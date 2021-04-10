# coding:utf-8
"""
Time: 2021/3/6 21:52
Author: eightyninth
File: train.py
"""
import numpy as np
import torch
import re
import time
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

is_resume = False
lr = 1e-3
resume_path = "./save_model_1/hde_net_491.pth"
start_epoch = 0
epochs = 50

train_step = 0

mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
    # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.3),  # 随机修改图像颜色空间
    transforms.RandomPerspective(),  # 透视变换
    transforms.ToTensor(),  # [0-255] -> [0,1]
    transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)

val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
        transforms.ToTensor(),  # [0-255] -> [0,1]
        transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)

train_data = hde_word_dataset("./dataset/picture_1", is_train=True, transform=train_transforms)
val_data = hde_word_dataset("./dataset/picture_1", is_train=False, transform=val_transforms)

hde_dict_train = train_data.get_dict()
hde_dict_val = val_data.get_dict()

train_dataloader = DataLoader(dataset=train_data, batch_size=15, num_workers=4, shuffle=True, pin_memory=True,
                              drop_last=True)  # , collate_fn=util.collate_fn)
val_dataloader = DataLoader(dataset=val_data, batch_size=15, num_workers=4, shuffle=False, pin_memory=True,
                            drop_last=True)  # , collate_fn=util.collate_fn)

# 部首有663个
model = HDENet(663, 256, "resnet34")

device = None

if torch.cuda.is_available():
    cudnn.benchmark = True
    device = 'cuda'
else:
    device = "cpu"

model.to(device)

# resume
if is_resume:
    util.load_model(model, resume_path)
    start_epoch = int(re.sub("\D", "", resume_path)) + 1

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

for epoch in range(start_epoch, start_epoch + epochs):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    scheduler.step()

    model.train()

    print('Epoch: {} : lr={}.'.format(epoch, scheduler.get_lr()))

    for i, (img, img_char) in enumerate(iter(train_dataloader)):

        train_step += 1

        hde_arr = np.array([v for k, v in hde_dict_train.items()])
        hde_arr = torch.from_numpy(hde_arr).type(torch.float32)

        img, hde_arr = img.to(device), hde_arr.to(device)

        hde_distance = model(img, hde_arr)

        # loss
        hde_cur = []
        for char in img_char:
            for j, h_a in enumerate(hde_arr.cpu().numpy()):
                if (h_a == hde_dict_train[char].astype(np.float32)).all():
                    hde_cur.append(j)
        # print(hde_cur)
        loss = util.loss_MCPL(hde_distance, hde_cur)

        acc = util.acc_cal(hde_distance, hde_cur)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc)

        print('\r({:d} / {:d})  Train Loss: {} Train Accuracy {}'.format(i, len(train_dataloader), loss.item(), acc), end="", flush=True)
        # time.sleep(0.001)
    with open("result_1.txt", "a+") as f:
        print('({:d} / {:d})  train Loss: {:.4f} train Accuracy {}'.format(epoch, start_epoch + epochs, np.mean(train_loss),
                                                                       np.mean(train_acc)), file=f, flush=True)
    print('\n({:d} / {:d})  Train Loss: {:.4f} Train Accuracy {}'.format(epoch, start_epoch + epochs, np.mean(train_loss), np.mean(train_acc)))


    for i, (img, img_char) in enumerate(val_dataloader):
        train_step += 1

        hde_arr = np.array([v for k, v in hde_dict_train.items()])
        hde_arr = torch.from_numpy(hde_arr).type(torch.float32)

        img, hde_arr = img.to(device), hde_arr.to(device)

        hde_distance = model(img, hde_arr)

        # loss
        hde_cur = []
        for char in img_char:
            for j, h_a in enumerate(hde_arr.cpu().numpy()):
                if (h_a == hde_dict_train[char].astype(np.float32)).all():
                    hde_cur.append(j)
        loss = util.loss_MCPL(hde_distance, hde_cur)

        acc = util.acc_cal(hde_distance, hde_cur)

        test_loss.append(loss.item())
        test_acc.append(acc)

    with open("result_1.txt","a+") as f:
        print('({:d} / {:d})  val Loss: {:.4f} val Accuracy {}'.format(epoch, start_epoch + epochs, np.mean(test_loss),
                                                                       np.mean(test_acc)), file=f, flush=True)
    print('({:d} / {:d})  val Loss: {:.4f} val Accuracy {}'.format(epoch, start_epoch + epochs, np.mean(test_loss), np.mean(test_acc)))

    util.save_model(model, epoch, scheduler.get_lr(), optimizer)
