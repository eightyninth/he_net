# coding:utf-8
"""
Time: 2021/3/22 9:38
Author: eightyninth
File: util.py
"""
import os

import torch
import numpy as np


def cos_distance(a, b):
    return 1 - torch.cosine_similarity(a, b, dim=-1)

def loss_MCPL(total_a,b):
    total = []
    for (t_a, t_b) in zip(total_a, b):
        t = 0.
        for a in t_a:
            t += torch.exp(a - t_a[t_b])
        total.append(torch.log(t))
    return torch.mean(torch.cat(total))

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

def load_model(model, model_path):
    print("Loading from {}.".format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])

def acc_cal(hde_distance, hde_cur):
    acc = 0
    for (h_d, h_c) in zip(hde_distance, hde_cur):
        hde_max_index = torch.argmax(torch.cat(h_d)).cpu().numpy()
        if hde_max_index == h_c:
            acc += 1

    return acc / len(hde_distance)

# def collate_fn(batch):
#
#     return batch
