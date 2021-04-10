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


def loss_MCPL(total_a, b):
    total = []
    # total_1 = []
    for (t_a, t_b) in zip(total_a, b):
        # t = 0.
        t_total = torch.log(sum(torch.exp(t_a - t_a[t_b])))
        total.append(t_total)
        # for a in t_a:
        #     t += torch.exp(a - t_a[t_b])
        # total.append(torch.log(t))

    # result = torch.mean(torch.Tensor(total)).requires_grad_(True)
    # result2 = sum(total) / len(total)
    # result1 = torch.mean(total_1)
    # return torch.mean(torch.Tensor(total)).requires_grad_(True)
    return sum(total) / len(total)


def save_model(model, epoch, lr, optimzer):
    save_dir = os.path.join("./save_model_1")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'hde_net_{}.pth'.format(epoch))
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
        hde_max_index = h_d.argmax().cpu().numpy()
        if hde_max_index == h_c:
            acc += 1

    return acc / hde_distance.shape[0]

# def collate_fn(batch):
#
#     return batch
