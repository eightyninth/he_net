# coding:utf-8
"""
Time: 2021/3/6 21:52
Author: eightyninth
File: train.py
"""
import torch
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset_word import hde_word_dataset
from net.hde_net import HDENet

lr = 1e-4

train_step = 0

mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]

train_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

train_data = hde_word_dataset("./picture", is_train=True, transform=train_transforms)
val_data = hde_word_dataset("./picture", is_train=False, transform=train_transforms)
train_dataloader= DataLoader(dataset=train_data, batch_size=6, num_workers=4, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=3, num_workers=1, shuffle=True, pin_memory=True)

model = HDENet(670, "resnet50")

if torch.cuda.is_available():
    cudnn.benchmark = True
    model = model.to("cuda")
else:
    model = model.to("cpu")

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=00.9)

for epoch in range(100):
    scheduler.step()

    model.train()

    for i, (img, img_hde) in enumerate(train_dataloader):

        train_step += 1

        img, img_hde =