import torch
import torch.nn as nn
import numpy as np
import random

from methods.StyleAdv_ViT_protonet import ProtoNet
from methods.meta_template_StyleAdv_RN_GNN import MetaTemplate
from methods.tool_func import *


class StyleAdvViT(MetaTemplate):
    maml = False

    def __init__(self, vit_backbone, n_way, n_support, tf_path=None):
        # MetaTemplate 초기화를 위해 dummy model_func 사용
        dummy_model_func = lambda **kwargs: vit_backbone
        super(StyleAdvViT, self).__init__(dummy_model_func, n_way, n_support, tf_path=tf_path)

        # ViT 백본과 ProtoNet 설정
        self.feature = vit_backbone
        self.feat_dim = 384  # ViT-Small의 feature dimension

        # ProtoNet 초기화
        self.protonet = ProtoNet(vit_backbone)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # for global classifier
        self.method = 'ProtoNet'
        self.classifier = nn.Linear(self.feat_dim, 64)

    def cuda(self):
        self.feature.cuda()
        self.protonet.cuda()
        self.classifier.cuda()
        return self

    def set_forward(self, x, is_feature=False):
        x = x.cuda()

        if is_feature:
            # x는 이미 feature 형태: (n_way, n_support + n_query, feat_dim)
            z_all = x
        else:
            # x를 feature로 변환: (n_way, n_support + n_query, 3, 224, 224)
            batch_size = x.size(0) * x.size(1)
            x_reshaped = x.view(batch_size, *x.size()[2:])
            z_all = self.feature(x_reshaped)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        # ProtoNet forward
        z_support = z_all[:, :self.n_support]  # (n_way, n_support, feat_dim)
        z_query = z_all[:, self.n_support:]  # (n_way, n_query, feat_dim)

        # ProtoNet을 위한 reshaping
        supp_x = z_support.unsqueeze(0)  # (1, n_way, n_support, feat_dim)
        supp_y = torch.arange(self.n_way).unsqueeze(0).repeat(1, self.n_support).view(1, -1)  # (1, n_way*n_support)
        query_x = z_query.unsqueeze(0)  # (1, n_way, n_query, feat_dim)

        # ProtoNet forward (feature는 이미 계산됨)
        import torch.nn.functional as F

        supp_f = supp_x.view(1, -1, supp_x.size(-1))  # (1, n_way*n_support, feat_dim)
        query_f = query_x.view(1, -1, query_x.size(-1))  # (1, n_way*n_query, feat_dim)

        supp_y_1hot = F.one_hot(supp_y.view(-1), self.n_way).transpose(0, 1).unsqueeze(
            0).float()  # (1, n_way, n_way*n_support)

        # 프로토타입 계산
        prototypes = torch.bmm(supp_y_1hot, supp_f)  # (1, n_way, feat_dim)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)

        # 코사인 분류기 (ProtoNet의 cos_classifier 사용)
        scores = self.protonet.cos_classifier(prototypes, query_f)  # (1, n_way*n_query, n_way)
        scores = scores.view(-1, self.n_way)  # (n_way*n_query, n_way)

        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)
        return scores, loss

    def adversarial_attack_Incre(self, x_ori, y_ori, epsilon_list):
        x_ori = x_ori.cuda()
        y_ori = y_ori.cuda()
        x_size = x_ori.size()
        x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        y_ori = y_ori.view(x_size[0] * x_size[1])

        # if not adv, set default = 'None'
        adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
        adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
        adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

        # forward and set the grad = True
        blocklist = 'block123'

        # Block 1 공격
        if ('1' in blocklist and epsilon_list[0] != 0):
            x_ori_block1 = self.feature.forward_block1(x_ori)
            x_ori_block1_cls, x_ori_block1_P = self.preprocessing(x_ori_block1)
            feat_size_block1 = x_ori_block1_P.size()

            ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1_P)
            ori_style_mean_block1 = torch.nn.Parameter(ori_style_mean_block1)
            ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
            ori_style_mean_block1.requires_grad_()
            ori_style_std_block1.requires_grad_()

            x_normalized_block1 = (x_ori_block1_P - ori_style_mean_block1.detach().expand(
                feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
            x_ori_block1_P = x_normalized_block1 * ori_style_std_block1.expand(
                feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
            x_ori_block1 = self.postprocessing(x_ori_block1_cls, x_ori_block1_P)

            # 나머지 블록들 통과
            x_ori_block2 = self.feature.forward_block2(x_ori_block1)
            x_ori_block3 = self.feature.forward_block3(x_ori_block2)
            x_ori_block4 = self.feature.forward_block4(x_ori_block3)
            x_ori_fea = self.feature.forward_rest(x_ori_block4)
            x_ori_output = self.classifier.forward(x_ori_fea)

            ori_loss = self.loss_fn(x_ori_output, y_ori)
            self.feature.zero_grad()
            self.classifier.zero_grad()
            ori_loss.backward()

            grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
            grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()

            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]

            adv_style_mean_block1 = fgsm_attack(ori_style_mean_block1, epsilon, grad_ori_style_mean_block1)
            adv_style_std_block1 = fgsm_attack(ori_style_std_block1, epsilon, grad_ori_style_std_block1)

        # Block 2, 3도 유사하게 구현 (생략)

        self.feature.zero_grad()
        self.classifier.zero_grad()

        return adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3

    def preprocessing(self, x_fea):
        """ViT feature preprocessing: [B, 197, 384] -> cls_fea [B, 1, 384], patch_fea [B, 384, 14, 14]"""
        B, num, dim = x_fea.size()
        x_cls_fea = x_fea[:, :1, :]
        x_patch_fea = x_fea[:, 1:, :]
        x_patch_fea = x_patch_fea.contiguous().view(B, dim, num - 1).view(B, dim, 14, 14)
        return x_cls_fea, x_patch_fea

    def postprocessing(self, x_cls_fea, x_patch_fea):
        """ViT feature postprocessing: cls_fea [B, 1, 384], patch_fea [B, 384, 14, 14] -> [B, 197, 384]"""
        B, num, dim = x_patch_fea.size()[0], x_patch_fea.size()[2] * x_patch_fea.size()[3] + 1, x_patch_fea.size()[1]
        x_patch_fea = x_patch_fea.contiguous().view(B, dim, num - 1).view(B, num - 1, dim)
        x_fea = torch.cat((x_cls_fea, x_patch_fea), 1)
        return x_fea

    def changeNewAdvStyle_ViT(self, vit_fea, new_styleAug_mean, new_styleAug_std, p_thred):
        """ViT용 스타일 공격 적용"""
        if new_styleAug_mean == 'None':
            return vit_fea

        p = np.random.uniform()
        if p < p_thred:
            return vit_fea

        cls_fea, input_fea = self.preprocessing(vit_fea)
        feat_size = input_fea.size()
        ori_style_mean, ori_style_std = calc_mean_std(input_fea)

        normalized_fea = (input_fea - ori_style_mean.expand(feat_size)) / ori_style_std.expand(feat_size)
        styleAug_fea = normalized_fea * new_styleAug_std.expand(feat_size) + new_styleAug_mean.expand(feat_size)
        styleAug_fea_vit = self.postprocessing(cls_fea, styleAug_fea)
        return styleAug_fea_vit

    def set_statues_of_modules(self, flag):
        if flag == 'eval':
            self.feature.eval()
            self.protonet.eval()
            self.classifier.eval()
        elif flag == 'train':
            self.feature.train()
            self.protonet.train()
            self.classifier.train()
        return

    def set_forward_loss_StyAdv(self, x_ori, global_y, epsilon_list):
        # StyleAdv 공격 수행
        x_adv = x_ori

        # 1. styleAdv
        self.set_statues_of_modules('eval')
        adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = self.adversarial_attack_Incre(
            x_ori, global_y, epsilon_list)

        self.feature.zero_grad()
        self.classifier.zero_grad()

        # 2. forward and get loss
        self.set_statues_of_modules('train')

        # FSL용 라벨
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        # original forward
        x_ori = x_ori.cuda()
        x_size = x_ori.size()
        x_ori = x_ori.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])
        global_y = global_y.view(x_size[0] * x_size[1]).cuda()

        x_ori_fea = self.feature.forward(x_ori)

        # original classifier loss
        scores_cls_ori = self.classifier.forward(x_ori_fea)
        loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)

        # original FSL loss
        scores_fsl_ori = self.set_forward(x_ori.view(x_size), is_feature=False)
        loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

        # adversarial forward
        x_adv = x_adv.cuda()
        x_adv = x_adv.view(x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4])

        x_adv_block1 = self.feature.forward_block1(x_adv)
        x_adv_block1_newStyle = self.changeNewAdvStyle_ViT(x_adv_block1, adv_style_mean_block1, adv_style_std_block1,
                                                           p_thred=P_THRED)
        x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
        x_adv_block2_newStyle = self.changeNewAdvStyle_ViT(x_adv_block2, adv_style_mean_block2, adv_style_std_block2,
                                                           p_thred=P_THRED)
        x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
        x_adv_block3_newStyle = self.changeNewAdvStyle_ViT(x_adv_block3, adv_style_mean_block3, adv_style_std_block3,
                                                           p_thred=P_THRED)
        x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
        x_adv_fea = self.feature.forward_rest(x_adv_block4)

        # adversarial classifier loss
        scores_cls_adv = self.classifier.forward(x_adv_fea)
        loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)

        # adversarial FSL loss
        scores_fsl_adv = self.set_forward(x_adv.view(x_size), is_feature=False)
        loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

        return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv