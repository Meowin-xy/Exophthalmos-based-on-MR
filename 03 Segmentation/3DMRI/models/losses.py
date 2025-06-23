"""
@author: zhengyong Huang
Time: 2022-10-21
"""

import numpy as np
import torch
from torch import nn
from calendar import c
import os
import torch.nn.functional as F
import cv2
import SimpleITK as sitk
import scipy.ndimage as scip
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        # print("inter: ", intersect.item())
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        # union = input.sum(dim=axes) + target.sum(dim=axes)
        # print("union: ", union.item())
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        # print("dice loss: ", loss.mean())
        return loss.mean()


class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        # print("focal loss: ", loss.mean())
        return loss.mean()

class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)
        self.BinaryTverskyLoss = BinaryTverskyLossV2()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss



if __name__ == "__main__":
    x = torch.rand(1, 1, 144, 144, 144)
    tar = torch.rand(1, 1, 144, 144, 144)
    DF = Dice_and_FocalLoss()
    loss = DF(x, tar)
    print("loss: ", loss)