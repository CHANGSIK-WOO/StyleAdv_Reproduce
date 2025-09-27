import numpy as np
import torch
import torch.optim
import os
import random

from methods.load_ViT_models import get_model
from data.datamgr import SimpleDataManager, SetDataManager
from methods.StyleAdv_ViT_meta_template import StyleAdvViT

from options import parse_args, get_resume_file, load_warmup_state
from test_function_bscdfsl_benchmark import test_bestmodel_bscdfsl


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    # get optimizer and checkpoint path
    optimizer = torch.optim.SGD(
        [{'params': p for p in model.feature.parameters() if p.requires_grad},
         {'params': model.classifier.parameters(), 'lr': 0.001}],
        lr=5e-5,
        momentum=0.9,
        weight_decay=0,
    )

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # for validation
    max_acc = 0
    total_it = 0

    # start
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        total_it = model.train_loop(epoch, base_loader, optimizer, total_it)
        model.eval()

        acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        else:
            print("GG! best accuracy {:f}".format(max_acc))

        if (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


def record_test_result_bscdfsl(params):
    print('testing for bscdfsl')
    acc_file_path = os.path.join(params.checkpoint_dir, 'acc_bscdfsl.txt')
    acc_file = open(acc_file_path, 'w')
    epoch_id = -1
    print('epoch', epoch_id, 'EuroSAT:', file=acc_file)
    name = params.name
    n_shot = params.n_shot

    # EuroSAT 테스트만 실행 (RN 버전과 동일)
    test_bestmodel_bscdfsl(acc_file, name, 'EuroSAT', n_shot, epoch_id)

    acc_file.close()
    return


# --- main function ---
if __name__ == '__main__':
    # fix seed
    seed = 0
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parser argument
    params = parse_args('train')

    # output and tensorboard dir
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader (RN 스타일과 동일)
    print('\n--- prepare dataloader ---')
    print('  train with single seen domain {}'.format(params.dataset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.dataset, 'val.json')

    # model
    print('\n--- build model ---')
    image_size = 224

    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    # ViT 백본과 StyleAdv 모델 생성
    vit_backbone = get_model(backbone='vit_small', classifier='protonet', styleAdv=False).backbone
    model = StyleAdvViT(vit_backbone, tf_path=params.tf_dir, **train_few_shot_params)
    model = model.cuda()

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume != '':
        resume_file = get_resume_file('%s/checkpoints/%s' % (params.save_dir, params.resume), params.resume_epoch)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
        if params.warmup == 'gg3b0':
            raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
        # ViT의 경우 pre-trained weight가 이미 로드되어 있음
        print('  using pre-trained ViT weights')

    import time

    start = time.time()
    # training
    print('\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)
    end = time.time()
    print('Running time: %s Seconds: %s Min: %s Min per epoch' % (end - start, (end - start) / 60,
                                                                  (end - start) / 60 / params.stop_epoch))

    # testing bscdfsl (RN 버전과 동일)
    record_test_result_bscdfsl(params)