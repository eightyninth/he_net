# coding:utf-8
"""
Time: 2021/3/6 20:28
Author: eightyninth
File: hde_net.py
"""
from abc import ABC

import torch
import torch.nn as nn
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo

from utils import util

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module, ABC):
    def __init__(self, name="resnet34", pretrain=True):
        super().__init__()
        if name == "resnet50":
            base_net = resnet.resnet50(pretrained=False)

            if pretrain:
                print("load the {} weight from ./cache".format(name))
                base_net.load_state_dict(model_zoo.load_url(model_urls["resnet50"], model_dir="./cache"))
        if name == "resnet34":
            base_net = resnet.resnet34(pretrained=False)

            if pretrain:
                print("load the {} weight from ./cache".format(name))
                base_net.load_state_dict(model_zoo.load_url(model_urls["resnet34"], model_dir="./cache"))

        self.stage1 = nn.Sequential(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool
        )
        self.stage2 = base_net.layer1
        self.stage3 = base_net.layer2
        self.stage4 = base_net.layer3
        self.stage5 = base_net.layer4

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)

        return C5


class HDENet(nn.Module, ABC):
    def __init__(self, out_dim, backbone="resnet50"):
        super(HDENet, self).__init__()
        if backbone == "resnet34":
            self.backbone = ResNet("resnet34", True)
        if backbone == "resnet50":
            self.backbone = ResNet("resnet50", True)

        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.fc2 = nn.Linear(512, out_dim, bias=False)

        self.d_p = nn.Parameter(torch.FloatTensor(1))

    def forward(self, img, hde_arr):
        # feature and flatten
        feat = self.backbone(img)
        feat = feat.view(img.size()[0], -1)
        # transfer
        fc1 = self.fc1(feat)
        fc2 = self.fc2(fc1)
        # compatibility
        distance = util.cos_distance(fc2, hde_arr)
        hde_distance = - self.d_p * distance
        return hde_distance


if __name__ == "__main__":
    model = HDENet(234, "resnet34")
