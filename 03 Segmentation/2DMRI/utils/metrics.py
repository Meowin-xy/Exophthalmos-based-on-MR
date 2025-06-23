"""
@author: zhengyong Huang
Time: 2022-07-12
"""

import torch
import torch.nn.functional as F

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()
    # input = input.cpu().numpy()
    # target = target.cpu().numpy()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()

def dice_b(inputs, targets):
    '''
    input  : 1 2 144 144 144
    target : 1 144 144 144
    '''
    eps = 0.001
    N, C, D, H, W = inputs.size()  
    prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320
    prob = (prob > 0.5).float()
    targets = targets.long()
    t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
    t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320

    # 去掉背景
    prob = prob[:, 1:, :, :, :]
    t_one_hot = t_one_hot[:, 1:, :, :, :]
    
    iflat = prob.view(-1)
    tflat = t_one_hot.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps)

    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()
