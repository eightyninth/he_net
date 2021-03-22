# coding:utf-8
"""
Time: 2021/3/22 9:38
Author: eightyninth
File: util.py
"""
import os

import torch

def cos_distance(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a, b, dim=1)

def loss_MCPL(total_a,b):
    total = 0.0
    for a in total_a:
        total += torch.exp(a - b)

    return torch.log(total)

def save_model(model, epoch, lr, optimzer):
    save_dir = os.path.join("./save_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'hde_net_{}.pth'.format( epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)
