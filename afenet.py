import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





def Entropy(softmax_out):
    epsilon = 1e-5
    softmax_out = softmax_out.view(softmax_out.size()[0],2,-1)
    entropy = -torch.sum(softmax_out * torch.log(softmax_out + epsilon), dim=1)
    entropy = torch.mean(entropy, dim=1)
    return entropy


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=200.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()

        def deconv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.adv_1 = nn.Sequential(
            *deconv_block(128, 64),
            *deconv_block(64, 64),
            *deconv_block(64, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(32, 32),
            *deconv_block(32, 16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.adv_2 = nn.Sequential(
            *deconv_block(256, 128),
            *deconv_block(128, 128),
            *deconv_block(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(64, 32),
            *deconv_block(32, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.adv_3 = nn.Sequential(
            *deconv_block(512, 256),
            *deconv_block(256, 256),
            *deconv_block(256, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(128, 64),
            *deconv_block(64, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        in_chanel = x.size()[1]
        if in_chanel == 128:
            out = self.adv_1(x)
        elif in_chanel == 256:
            out = self.adv_2(x)
        elif in_chanel == 512:
            out = self.adv_3(x)
        else:
            out = None
            print('--------------adnet error--------------------------')
        return out

    def get_parameters(self):
        parameter_list = [{"params": self.parameters()}]
        return parameter_list


class AdversarialNetwork_L(nn.Module):
    def __init__(self):
        super(AdversarialNetwork_L, self).__init__()

        def deconv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.adv_1 = nn.Sequential(
            *deconv_block(256, 128),
            *deconv_block(128, 128),
            *deconv_block(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(64, 64),
            *deconv_block(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.adv_2 = nn.Sequential(
            *deconv_block(512, 256),
            *deconv_block(256, 256),
            *deconv_block(256, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(128, 64),
            *deconv_block(64, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.adv_3 = nn.Sequential(
            *deconv_block(1024, 512),
            *deconv_block(512, 512),
            *deconv_block(512, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1),
            *deconv_block(256, 128),
            *deconv_block(128, 128),
            nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        in_chanel = x.size()[1]
        if in_chanel == 256:
            out = self.adv_1(x)
        elif in_chanel == 512:
            out = self.adv_2(x)
        elif in_chanel == 1024:
            out = self.adv_3(x)
        else:
            out = None
            print('--------------adnet error--------------------------')
        return out

    def get_parameters(self):
        parameter_list = [{"params": self.parameters()}]
        return parameter_list


def CADA(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    feature = input_list[0]
    #input_list[1] = torch.cat((out_1, out_2), dim=0)
    softmax_output = input_list[1].detach()
    softmax_output = F.adaptive_max_pool2d(softmax_output,feature[0][0].size())
    # feature = up(feature)
    if random_layer is None:
        op_out = torch.matmul(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        op_out_1 = op_out.view(op_out.size(0),op_out.size(1)*op_out.size(2),op_out.size(3),op_out.size(4))
        ad_out = ad_net(op_out_1)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    ones_matrix = torch.ones((batch_size, 1, ad_out.size(2), ad_out.size(3))).float()
    zeros_matrix = torch.zeros((batch_size, 1, ad_out.size(2), ad_out.size(3))).float()
    dc_target = torch.cat((ones_matrix, zeros_matrix), dim=0).cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss()(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


class HybirdFusion_Block(nn.Module):
    def __init__(self, in_features, out_features, key = '1'):
        super(HybirdFusion_Block, self).__init__()

        def conv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.strategy = key
        self.layer_1 = nn.Sequential(
            *conv_block(in_features * 3, in_features),
            *conv_block(in_features , out_features),
            *conv_block(out_features, out_features),
        )
        self.layer_2 = nn.Sequential(
            *conv_block(in_features * 3, in_features),
            *conv_block(in_features, out_features),
            *conv_block(out_features, out_features),
            *conv_block(out_features, out_features),
        )
        self.layer_3 = nn.Sequential(
            *conv_block(in_features * 4, in_features * 2),
            *conv_block(in_features * 2, out_features),
            *conv_block(out_features, out_features),
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_features * 2, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x1, x2, xx):
        if self.strategy == '1':
            out_fusion = torch.cat((x1, x2, xx), dim=1)
            out = self.layer_1(out_fusion)
        elif self.strategy == '2':
            out_fusion = torch.cat((x1, x2, xx), dim=1)
            out = self.layer_2(out_fusion)
        elif self.strategy == '3':
            out_fusion = torch.cat((x1, x2, xx), dim=1)
            out = self.layer_3(out_fusion)
        elif self.strategy == '4':
            fusion_sum = torch.add(x1, x2)
            fusion_mul = torch.mul(x1, x2)
            fusion_dec = torch.abs(x1 - x2)
            out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_dec, xx), dim=1)
            out = self.layer_2(out_fusion)
        else:
            fusion_dec = torch.abs(x1 - x2)
            out_fusion = torch.cat((fusion_dec, xx),dim=1)
            out = self.layer_3(out_fusion)

        return out


class SE(nn.Module):
    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = torch.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)
        return out


class CA_Fusion(nn.Module):
    def __init__(self, in_features, out_features, last=False):
        super(CA_Fusion, self).__init__()

        def conv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.se = SE(in_features,4)
        if last:
            self.layer = nn.Sequential(
                *conv_block(in_features * 3, in_features),
                *conv_block(in_features, out_features),
            )
        else:
            self.layer = nn.Sequential(
                *conv_block(in_features * 3, in_features),
                *conv_block(in_features, out_features),
                *conv_block(out_features, out_features),
        )

    def forward(self, x1, x2, xx):
        weight = self.se(xx)
        # xx_ca = xx * weight
        x1_ca = x1 * weight
        x2_ca = x2 * weight
        out_fusion = torch.cat((xx, x1_ca, x2_ca), dim=1)
        out = self.layer(out_fusion)
        return out


class AFENet(nn.Module):

    def __init__(self, input_nbr, output_nbr):
        super(AFENet, self).__init__()

        def conv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.modal_1 = nn.Sequential(
            *conv_block(input_nbr, 32),
            *conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(32, 64),
            *conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 128),
            *conv_block(128, 128),
            *conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(128, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # modal_2
        self.modal_2 = nn.Sequential(
            *conv_block(input_nbr, 32),
            *conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(32, 64),
            *conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 128),
            *conv_block(128, 128),
            *conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(128, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # modal_3
        self.modal_3 = nn.Sequential(
            *conv_block(6, 32),
            *conv_block(32,32),
        )

        # fusion
        self.fu_1 = HybirdFusion_Block(32,64,key='1')
        self.down_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_2 = HybirdFusion_Block(64, 128, key='1')
        self.down_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_3 = HybirdFusion_Block(128, 256, key='2')
        self.down_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_4 = HybirdFusion_Block(256, 512, key='2')
        self.down_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_5 = HybirdFusion_Block(256, 512, key='3')

        def deconv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        # CD
        self.upconv_4 = nn.Sequential(
            *deconv_block(512, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, stride=2, output_padding=1)
        )
        self.deconv_4 = nn.Sequential(
            *deconv_block(1024, 512),
            *deconv_block(512, 512),
            *deconv_block(512, 256)
        )

        self.upconv_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_3 = nn.Sequential(
            *deconv_block(512, 256),
            *deconv_block(256, 256),
            *deconv_block(256, 128)
        )

        self.upconv_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_2 = nn.Sequential(
            *deconv_block(256, 128),
            *deconv_block(128, 128),
            *deconv_block(128, 64)
        )

        self.upconv_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_1 = nn.Sequential(
            *deconv_block(128, 64),
        )

        self.out = nn.ConvTranspose2d(64, output_nbr, kernel_size=3, padding=1)


    def forward(self, x1, x2):
        feats = [[]for i in range(2)]

        # modal_1
        feature_1_1 = self.modal_1[:7](x1)
        feature_2_1 = self.modal_1[:16](x1)
        feature_3_1 = self.modal_1[:29](x1)
        feature_4_1 = self.modal_1[:42](x1)
        feature_5_1 = self.modal_1(x1)

        # modal_2
        feature_1_2 = self.modal_2[:7](x2)
        feature_2_2 = self.modal_2[:16](x2)
        feature_3_2 = self.modal_2[:29](x2)
        feature_4_2 = self.modal_2[:42](x2)
        feature_5_2 = self.modal_2(x2)

        # modal_1+2
        xx = torch.cat((x1, x2), dim=1)
        feature_1 = self.modal_3(xx)

        for i in range(1,6):
            feats[0].append(locals()[f'feature_{i}_1'])
            feats[1].append(locals()[f'feature_{i}_2'])

        # fusion
        fusion_1 = self.fu_1(feature_1_1, feature_1_2, feature_1)
        fusion_1_d = self.down_2(fusion_1)
        fusion_2 = self.fu_2(feature_2_1, feature_2_2, fusion_1_d)
        fusion_2_d = self.down_3(fusion_2)
        fusion_3 = self.fu_3(feature_3_1, feature_3_2, fusion_2_d)
        fusion_3_d = self.down_4(fusion_3)
        fusion_4 = self.fu_4(feature_4_1, feature_4_2, fusion_3_d)
        fusion_4_d = self.down_5(fusion_4)
        fusion_5 = self.fu_5(feature_5_1, feature_5_2, fusion_4_d)

        # CD
        up_4 = self.upconv_4(torch.cat((feature_5_1,feature_5_2), dim=1))
        up_4 = torch.cat((up_4, fusion_4), dim=1)
        cd_4 = self.deconv_4(up_4)


        up_3 = self.upconv_3(cd_4)
        up_3 = torch.cat((up_3, fusion_3), dim=1)
        cd_3 = self.deconv_3(up_3)

        up_2 = self.upconv_2(cd_3)
        up_2 = torch.cat((up_2, fusion_2), dim=1)
        cd_2 = self.deconv_2(up_2)

        up_1 = self.upconv_1(cd_2)
        up_1 = torch.cat((up_1, fusion_1), dim=1)
        cd_1 = self.deconv_1(up_1)
        out_cd = self.out(cd_1)

        return out_cd, feats

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list



class AFENet_L(nn.Module):

    def __init__(self, input_nbr, output_nbr):
        super(AFENet_L, self).__init__()

        def conv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        self.modal_1 = nn.Sequential(
            *conv_block(input_nbr, 64),
            *conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 128),
            *conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(128, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(256, 512),
            *conv_block(512, 512),
            *conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # modal_2
        self.modal_2 = nn.Sequential(
            *conv_block(input_nbr, 64),
            *conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 128),
            *conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(128, 256),
            *conv_block(256, 256),
            *conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(256, 512),
            *conv_block(512, 512),
            *conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # modal_3
        self.modal_3 = nn.Sequential(
            *conv_block(6, 64),
            *conv_block(64,64),
        )

        # fusion
        self.fu_1 = HybirdFusion_Block(64,128,key='1')
        self.down_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_2 = HybirdFusion_Block(128, 256, key='1')
        self.down_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_3 = HybirdFusion_Block(256, 512, key='2')
        self.down_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_4 = HybirdFusion_Block(512, 1024, key='2')
        self.down_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fu_5 = HybirdFusion_Block(512, 1024, key='3')

        def deconv_block(in_features, out_features, normalize=True, dropout=True):
            layers = [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout2d(p=0.2))
            return layers

        # CD
        self.upconv_4 = nn.Sequential(
            *deconv_block(1024, 1024),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, padding=1, stride=2, output_padding=1)
        )
        self.deconv_4 = nn.Sequential(
            *deconv_block(2048, 1024),
            *deconv_block(1024, 1024),
            *deconv_block(1024, 512)
        )

        self.upconv_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_3 = nn.Sequential(
            *deconv_block(1024, 512),
            *deconv_block(512, 512),
            *deconv_block(512, 256)
        )

        self.upconv_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_2 = nn.Sequential(
            *deconv_block(512, 256),
            *deconv_block(256, 256),
            *deconv_block(256, 128)
        )

        self.upconv_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_1 = nn.Sequential(
            *deconv_block(256, 128),
            *deconv_block(128, 128),
            *deconv_block(128, 64),
        )

        self.out = nn.ConvTranspose2d(64, output_nbr, kernel_size=3, padding=1)


    def forward(self, x1, x2):
        feats = [[]for i in range(2)]

        # modal_1
        feature_1_1 = self.modal_1[:7](x1)
        feature_2_1 = self.modal_1[:16](x1)
        feature_3_1 = self.modal_1[:29](x1)
        feature_4_1 = self.modal_1[:42](x1)
        feature_5_1 = self.modal_1(x1)

        # modal_2
        feature_1_2 = self.modal_2[:7](x2)
        feature_2_2 = self.modal_2[:16](x2)
        feature_3_2 = self.modal_2[:29](x2)
        feature_4_2 = self.modal_2[:42](x2)
        feature_5_2 = self.modal_2(x2)

        # modal_1+2
        xx = torch.cat((x1, x2), dim=1)
        feature_1 = self.modal_3(xx)

        for i in range(1,6):
            feats[0].append(locals()[f'feature_{i}_1'])
            feats[1].append(locals()[f'feature_{i}_2'])

        # fusion
        fusion_1 = self.fu_1(feature_1_1, feature_1_2, feature_1)
        fusion_1_d = self.down_2(fusion_1)
        fusion_2 = self.fu_2(feature_2_1, feature_2_2, fusion_1_d)
        fusion_2_d = self.down_3(fusion_2)
        fusion_3 = self.fu_3(feature_3_1, feature_3_2, fusion_2_d)
        fusion_3_d = self.down_4(fusion_3)
        fusion_4 = self.fu_4(feature_4_1, feature_4_2, fusion_3_d)
        fusion_4_d = self.down_5(fusion_4)
        fusion_5 = self.fu_5(feature_5_1, feature_5_2, fusion_4_d)

        # CD
        up_4 = self.upconv_4(torch.cat((feature_5_1,feature_5_2), dim=1))
        up_4 = torch.cat((up_4, fusion_4), dim=1)
        cd_4 = self.deconv_4(up_4)


        up_3 = self.upconv_3(cd_4)
        up_3 = torch.cat((up_3, fusion_3), dim=1)
        cd_3 = self.deconv_3(up_3)

        up_2 = self.upconv_2(cd_3)
        up_2 = torch.cat((up_2, fusion_2), dim=1)
        cd_2 = self.deconv_2(up_2)

        up_1 = self.upconv_1(cd_2)
        up_1 = torch.cat((up_1, fusion_1), dim=1)
        cd_1 = self.deconv_1(up_1)
        out_cd = self.out(cd_1)

        return out_cd, feats

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list




if __name__ == '__main__':

    state_dict = torch.load('/home/yan/Documents/CD_Code/Work1/best_G_HTCD_MDD_0.95.pth')
    # 打印 state_dict 的键值
    for key in state_dict.keys():
        print(key)

    # x1 = torch.randn(4, 3, 256, 256).cuda()
    # x2 = torch.randn(4, 3, 256, 256).cuda()
    # model = AFENet(3,2).cuda()
    # out = model(x1, x2)



