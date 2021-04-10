# coding:utf-8
"""
Time: 2021/3/6 19:32
Author: eightyninth
File: dataset_word.py
"""
import csv
import os
import imghdr
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class hde_word_dataset(Dataset):
    def __init__(self, data_path="./hde", is_train=True, transform=None):
        super(hde_word_dataset, self).__init__()

        self.data_path = data_path
        self.is_train = is_train

        # 数据增强
        self.transform = transform

        # 编码存储文件位置
        word_path = os.path.join(self.data_path, "hde.csv")
        # 读取对应字符与编码
        with open(word_path, "r", encoding="gb18030") as wf:
            reader = csv.reader(wf)
            word_hde = [row for row in reader]

        self.hde_dict = {}  # {字符: 编码}
        self.imgs_train = []  # [字符, img_path]
        self.imgs_val = []  # [字符, img_path]

        for hde in word_hde:

            img_label = hde[0]  # 文字
            root_path = os.path.join(self.data_path, img_label.encode("utf-8").decode("utf-8"))
            imgs_path = os.listdir(root_path)  # 得到图像的存储位置

            if len(imgs_path) < 2:
                continue

            # 记录编码
            self.hde_dict.update({img_label: np.array([float(h) for h in hde[1:]])})

            # 记录图像路径
            for i in range(len(imgs_path)):
                img_path = os.path.join(root_path, imgs_path[i].encode("utf-8").decode("utf-8"))
                img_list = [img_label, img_path]  # [字符, img_path]
                # 训练集
                if self.is_train and i < len(imgs_path):
                    self.imgs_train.append(img_list)
                # 测试集
                elif not self.is_train and len(imgs_path) - 1 <= i < len(imgs_path):
                    self.imgs_val.append(img_list)
                else:
                    continue

        pass    # test

    def get_dict(self):
        return self.hde_dict

    def __len__(self):
        if self.is_train:
            img_num = len(self.imgs_train)
        else:
            img_num = len(self.imgs_val)
        return img_num

    def __getitem__(self, index):
        if self.is_train:
            img_path = self.imgs_train[index][-1]
            char = self.imgs_train[index][0]
        else:
            img_path = self.imgs_val[index][-1]
            char = self.imgs_val[index][0]

        # jpg or png
        img = Image.open(img_path)

        if len(img.size) != 3:
            img = img.convert("RGB")

        img = np.array(img)

        # padding to square
        h, w, c = img.shape
        img_pad = np.zeros((h, h, c), dtype=img.dtype) if h > w else np.zeros((w, w, c), dtype=img.dtype)
        img_pad[:h, :w] = img
        if self.transform is not None:
            img_pad = self.transform(img_pad)

        return img_pad, char


if __name__ == "__main__":
    # pytorch 预定义均值与方差
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]

    # 训练时数据增强
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.3),  # 随机修改图像颜色空间
        transforms.RandomPerspective(),  # 透视变换
        transforms.ToTensor(),  # [0-255] -> [0,1]
        transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)
    # 验证时数据增强
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
        transforms.ToTensor(),  # [0-255] -> [0,1]
        transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)

    dataset_train = hde_word_dataset("./picture", is_train=True, transform=train_transforms)
    dataset_val = hde_word_dataset("./picture", is_train=False, transform=val_transforms)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)

    for i, (img, char) in enumerate(iter(dataloader_train)):
        # print(img)
        img = img[0].numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()

        pass
