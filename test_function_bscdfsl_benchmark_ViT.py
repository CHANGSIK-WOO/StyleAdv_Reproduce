import torch
import os
import h5py
import random
import numpy as np
import data.feature_loader as feat_loader
from data.datamgr import SimpleDataManager
from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from methods.load_ViT_models import load_ViTsmall
from methods.protonet import ProtoNet
from methods.StyleAdv_ViT_meta_template import StyleAdvViT
from options import get_best_file, get_assigned_file


# extract and save image features
def save_features(model, data_loader, featurefile):
    f = h5py.File(featurefile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feats = model(x)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


# evaluate using features
def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


def test_bestmodel_bscdfsl_ViT(acc_file, name, dataset, n_shot, save_epoch=-1):
    # 직접 params 객체 생성 (parse_args 사용하지 않음)
    class TestParams:
        def __init__(self):
            self.n_shot = n_shot
            self.dataset = dataset
            self.name = name
            self.save_epoch = save_epoch  # -1 = best
            self.data_dir = '/data/changsik/cdfsl-benchmark/filelists'
            self.save_dir = './output'
            self.test_n_way = 5
            self.split = 'novel'
            # 실제 모델 경로 설정
            self.checkpoint_dir = f'./output/{name}'
            self.model_path = f'./output/{name}/best.pth'

    params = TestParams()
    print(
        'Testing! {} shots on {} dataset with {} epochs of {}'.format(params.n_shot, params.dataset, params.save_epoch,
                                                                      params.name))
    remove_featurefile = True

    print('\nStage 1: saving features')
    # dataset
    print('  build dataset')
    image_size = 224
    split = params.split
    if (params.dataset in ["miniImagenet", "cub", "cars", "places", "plantae"]):
        loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
        print('load file:', loadfile)
        datamgr = SimpleDataManager(image_size, batch_size=64)
        data_loader = datamgr.get_data_loader(loadfile, aug=False)

    else:
        if params.dataset in ["ISIC"]:
            datamgr = ISIC_few_shot.SimpleDataManager(image_size, batch_size=64)
            data_loader = datamgr.get_data_loader(aug=False)

        elif params.dataset in ["EuroSAT"]:
            datamgr = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size=64)
            data_loader = datamgr.get_data_loader(aug=False)

        elif params.dataset in ["CropDisease"]:
            datamgr = CropDisease_few_shot.SimpleDataManager(image_size, batch_size=64)
            data_loader = datamgr.get_data_loader(aug=False)

        elif params.dataset in ["ChestX"]:
            datamgr = Chest_few_shot.SimpleDataManager(image_size, batch_size=64)
            data_loader = datamgr.get_data_loader(aug=False)

    print('  build feature encoder')
    # feature encoder - ViT 백본 사용
    modelfile = params.model_path
    if not os.path.exists(modelfile):
        print(f"Model file not found: {modelfile}")
        raise FileNotFoundError(f"Model file not found: {modelfile}")

    # ViT 모델 로드
    vit_backbone = load_ViTsmall()
    vit_backbone = vit_backbone.cuda()

    # 체크포인트에서 ViT 백본 가중치 로드
    tmp = torch.load(modelfile)

    # 키 확인 및 적절한 state 추출
    print(f'Available keys in checkpoint: {list(tmp.keys())}')

    # 다양한 키 패턴 시도
    state = None
    possible_keys = ['model', 'state', 'model_state']
    for key in possible_keys:
        if key in tmp:
            state = tmp[key]
            print(f'Using checkpoint key: {key}')
            break

    if state is None:
        raise KeyError(f"No valid model state found in checkpoint. Available keys: {list(tmp.keys())}")

    state_keys = list(state.keys())
    print('state_keys:', len(state_keys))

    # ViT feature encoder를 위한 state dict 준비
    feature_state = {}
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.", "")
            feature_state[newkey] = state[key]

    print('feature state keys:', len(feature_state))
    vit_backbone.load_state_dict(feature_state, strict=False)
    vit_backbone.eval()

    # save feature file
    print('  extract and save features...')
    if params.save_epoch != -1:
        featurefile = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"),
                                   split + "_" + str(params.save_epoch) + ".hdf5")
    else:
        featurefile = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")
    dirname = os.path.dirname(featurefile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(vit_backbone, data_loader, featurefile)

    print('\nStage 2: evaluate')
    acc_all = []
    iter_num = 1000
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    # ViT 기반 StyleAdv 모델
    print('  build ViT-based model')
    vit_backbone_eval = load_ViTsmall()
    model = StyleAdvViT(vit_backbone_eval, **few_shot_params)
    model = model.cuda()
    model.eval()

    # load model
    modelfile = params.model_path
    if modelfile is not None and os.path.exists(modelfile):
        tmp = torch.load(modelfile)

        # 모델 state 로드를 위한 키 확인
        print(f'Loading model from checkpoint keys: {list(tmp.keys())}')

        # 가능한 키들로 시도
        model_loaded = False
        possible_model_keys = ['model', 'state', 'model_state']

        for key in possible_model_keys:
            if key in tmp:
                try:
                    model.load_state_dict(tmp[key], strict=False)
                    print(f'Successfully loaded model using key: {key}')
                    model_loaded = True
                    break
                except Exception as e:
                    print(f'Failed to load model with key {key}: {e}')
                    continue

        if not model_loaded:
            raise KeyError(
                f"Could not load model from any of the keys: {possible_model_keys}. Available keys: {list(tmp.keys())}")

    # load feature file
    print('  load saved feature file')
    cl_data_file = feat_loader.init_loader(featurefile)

    # start evaluate
    print('  evaluate')
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
        acc_all.append(acc)

    # statics
    print('  get statics')
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('  %s %d test iterations: Acc = %4.2f%% +- %4.2f%%' % (params.dataset, iter_num, acc_mean,
                                                                 1.96 * acc_std / np.sqrt(iter_num)))
    print('  %s %d test iterations: Acc = %4.2f%% +- %4.2f%%' % (params.dataset, iter_num, acc_mean,
                                                                 1.96 * acc_std / np.sqrt(iter_num)), file=acc_file)

    # remove feature files [optional]
    if remove_featurefile:
        os.remove(featurefile)