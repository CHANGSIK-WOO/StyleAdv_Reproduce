import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args

from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

from methods.load_ViT_models import load_ViTsmall
from methods.protonet import ProtoNet

PMF_metatrained = True
FINAL_FEAT_DIM = 384


def load_model():
    """메타 트레인된 ViT 모델 로드 (파인튜닝 없이)"""
    vit_model = load_ViTsmall()
    model = ProtoNet(vit_model)

    if PMF_metatrained:
        # 학습된 모델 경로 - 사용자의 실제 경로로 수정 필요
        pmf_pretrained_ckp = 'output/ViT_5WAY_1SHOTS/best.pth'
        state_pmf = torch.load(pmf_pretrained_ckp)['model']

        # 키 변환 (feature. -> backbone.)
        state_new = state_pmf
        state_keys = list(state_pmf.keys())
        for i, key in enumerate(state_keys):
            if 'feature.' in key:
                newkey = key.replace("feature.", "backbone.")
                state_new[newkey] = state_pmf.pop(key)
            if 'classifier.' in key:
                state_new.pop(key)
            else:
                pass
        model.load_state_dict(state_new)

    model.eval().cuda()  # 파인튜닝 안 하므로 eval 모드
    return model


def set_forward_ViTProtonet(model, x):
    """ViT ProtoNet forward 함수"""
    n_way = x.size()[0]
    n_query = 15
    n_support = x.size()[1] - n_query

    SupportTensor = x[:, :n_support, :, :, :]
    QueryTensor = x[:, n_support:, :, :, :]
    SupportLabel = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
    QueryLabel = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()

    SupportTensor = SupportTensor.contiguous().view(-1, n_way * n_support, 3, 224, 224)
    QueryTensor = QueryTensor.contiguous().view(-1, n_way * n_query, 3, 224, 224)
    SupportLabel = SupportLabel.contiguous().view(-1, n_way * n_support)
    QueryLabel = QueryLabel.contiguous().view(-1, n_way * n_query)

    output = model(SupportTensor, SupportLabel, QueryTensor)
    output = output.view(n_way * n_query, n_way)
    return output


def direct_test(novel_loader, n_way=5, n_support=5):
    """파인튜닝 없이 바로 테스트"""
    iter_num = len(novel_loader)
    acc_all = []

    # 모델 한 번만 로드 (파인튜닝 안 하므로)
    model = load_model()

    for ti, (x, _) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        x = x.cuda()

        # 직접 추론 (파인튜닝 없이)
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)

        with torch.no_grad():
            scores = set_forward_ViTProtonet(model, x)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            acc = top1_correct * 100. / (n_way * n_query)
            acc_all.append(acc)

        del scores, topk_labels
        torch.cuda.empty_cache()

        if (ti % 50 == 0):
            print('Task %d : %4.2f%%, mean Acc: %4.2f' % (ti, acc, np.mean(np.array(acc_all))))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Direct Test (No Fine-tuning) Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    return acc_mean


def run_single_testset(params):
    """단일 데이터셋에 대해 파인튜닝 없이 테스트"""
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_size = 224
    iter_num = 1000
    n_query = 15

    print('Loading target dataset!:', params.testset)

    if params.testset in ['cub', 'cars', 'places', 'plantae']:
        novel_file = os.path.join(params.data_dir, params.testset, 'novel.json')
        datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot,
                                 n_eposide=iter_num)
        novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    else:
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        if params.testset in ["ISIC"]:
            datamgr = ISIC_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug=False)

        elif params.testset in ["EuroSAT"]:
            datamgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query,
                                                      **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug=False)

        elif params.testset in ["CropDisease"]:
            datamgr = CropDisease_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query,
                                                          **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug=False)

        elif params.testset in ["ChestX"]:
            datamgr = Chest_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug=False)

    # 파인튜닝 없이 바로 테스트
    acc = direct_test(novel_loader, n_way=params.test_n_way, n_support=params.n_shot)
    return acc


if __name__ == '__main__':
    params = parse_args(script='train')

    print("=" * 60)
    print("Direct Testing (No Fine-tuning) StyleAdv ViT")
    print("=" * 60)

    results = []

    # 모든 BS-CDFSL 데이터셋에 대해 파인튜닝 없이 테스트
    for tmp_testset in ['ISIC', 'EuroSAT', 'CropDisease', 'ChestX']:
        params.testset = tmp_testset
        print(f"\n{'=' * 50}")
        print(f"Direct Testing on {tmp_testset} (No Fine-tuning)")
        print(f"{'=' * 50}")

        acc = run_single_testset(params)
        results.append(f"{tmp_testset}: {acc:.2f}%")

        print(f"Result for {tmp_testset}: {acc:.2f}%")
        print("-" * 50)

    # 최종 결과 요약
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS - Direct Testing (No Fine-tuning)")
    print(f"{'=' * 60}")
    for result in results:
        print(result)
    print(f"{'=' * 60}")