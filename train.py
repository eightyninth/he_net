# coding:utf-8
"""
Time: 2021/3/6 21:52
Author: eightyninth
File: train.py
"""
import numpy as np
import torch
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset_word import hde_word_dataset
from net.hde_net import HDENet
from utils import util

lr = 1e-4

train_step = 0

mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),  # 把图像大小resize 到一个尺寸上
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.3),  # 随机修改图像颜色空间
    transforms.RandomPerspective(),  # 透视变换
    transforms.ToTensor(),  # [0-255] -> [0,1]
    transforms.Normalize(mean=mean, std=stdv)])  # 标准化 N ~ (0, 1)

train_data = hde_word_dataset("./picture", is_train=True, transform=train_transforms)
val_data = hde_word_dataset("./picture", is_train=False, transform=train_transforms)
train_dataloader= DataLoader(dataset=train_data, batch_size=6, num_workers=4, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=3, num_workers=1, shuffle=True, pin_memory=True)

# 部首有663个
model = HDENet(663, "resnet50")

device = None

if torch.cuda.is_available():
    cudnn.benchmark = True
    device = 'cuda'
else:
    device = "cpu"

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

train_loss = []
test_loss = []
test_acc = []


for epoch in range(100):
    scheduler.step()

    model.train()

    print('Epoch: {} : lr={}.'.format(epoch, scheduler.get_lr()))

    for i, (img, img_char, hde_dict) in enumerate(train_dataloader):

        train_step += 1

        hde_arr = torch.from_numpy(np.array([v for k, v in hde_dict.keys()]))

        img, hde_arr = img.to(device), hde_arr.to(device)

        hde_distance = model(img, hde_arr)

        # loss
        loss = util.loss_MCPL(hde_distance, hde_distance[hde_distance == hde_arr[hde_dict[img_char]]])

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss.append(loss)

    for i, (img, img_char, hde_dict) in enumerate(val_dataloader):
        train_step += 1

        hde_arr = torch.from_numpy(np.array([v for k, v in hde_dict.keys()]))

        img, hde_arr = img.to(device), hde_arr.to(device)

        hde_distance = model(img, hde_arr)

        # loss
        loss = util.loss_MCPL(hde_distance, hde_distance[hde_distance == hde_arr[hde_dict[img_char]]])

        test_loss.append(loss)

        print('({:d} / {:d})  Loss: {:.4f}'.format(i, len(train_dataloader), loss.item()))

        if epoch % 10 == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
