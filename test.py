# coding:utf-8
"""
Time: 2021/6/21 11:01
Author: eightyninth
File: test.py
"""
import numpy as np
import torch
import csv
from PIL import Image
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset_word import hde_word_dataset
from net.hde_net import HDENet
from utils import util

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# 加载 hde.csv
def load_hde(path, device):
    hde_dict = {}
    word_path = os.path.join(path, "hde.csv")

    # 读取对应字符与编码
    with open(word_path, "r", encoding="gb18030") as wf:
        reader = csv.reader(wf)
        for row in reader:
            hde_dict.update({row[0]: np.array([float(r) for r in row[1:]])})

    hde_arr = np.array([v for k, v in hde_dict.items()])
    hde_arr = torch.from_numpy(hde_arr).type(torch.float32)

    return hde_dict, hde_arr.to(device)

# 处理图片
def load_img(path, device, transforms=None):
    img = Image.open(path)

    if len(img.size) != 3:
        img = img.convert("RGB")

    img = np.array(img)

    # padding to square
    h, w, c = img.shape
    img_pad = np.zeros((h, h, c), dtype=img.dtype) if h > w else np.zeros((w, w, c), dtype=img.dtype)
    img_pad[:h, :w] = img

    if transforms is not None:
        img_pad = transforms(img_pad)

    return img_pad.to(device)

# 网络使用
def load_net(img, hde_arr, model, weight_path, device):

    model.to(device)
    util.load_model(model, weight_path)

    hde_distance = model(img, hde_arr)

    return hde_distance

# 结果后处理
def post_process(hde_dict, hde_arr, hde_dis):
    hde_max_index = hde_dis.argmax().cpu().numpy()
    hde_pred = hde_arr[hde_max_index]
    for k, v in hde_dict:
        if (hde_pred == v.astype(np.float32)).all():
            return k

    return "other"

if __name__ == "__main__":
    # pytorch 预定义均值与方差
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]

    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
        transforms.ToTensor(),  # [0-255] -> [0,1]
        transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = 'cuda'
    else:
        device = "cpu"

    # 部首有663个
    model = HDENet(663, 256, "resnet34")

    weight_path = ""
    hde_path = "./dataset/hde.csv"
    img_path = input("image_path: ")

    hde_dict, hde_arr = load_hde(hde_path, device)
    img = load_img(img_path, device, transforms=transforms)
    hde_dis = load_net(img, hde_arr, model, weight_path, device)
    char = post_process(hde_dict, hde_arr, hde_dis)

    print("当前图片对应的文字是: {}".format(char))
    pass