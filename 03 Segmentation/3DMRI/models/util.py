"""
@author: zhengyong Huang
Time: 2022-07-12
"""

import os
import os.path as osp
from datetime import datetime
import socket
import torch
import torch.nn.functional as F
import pdb
import SimpleITK as sitk
import numpy as np
import cv2

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def add_param_histogram(writer, net, global_step, pre=''):
    # for tensorboardX
    for name, param in net.named_parameters():
        # print (type(param.grad.data))
        if param.grad is not None:
            writer.add_histogram(pre + '/' + name, param.data.clone().cpu().numpy(), global_step, bins='doane')
            writer.add_histogram(pre + '/' + name + '.grad', param.grad.data.clone().cpu().numpy(), global_step, bins='doane')

def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()

def get_output_dir(args, run_name):
    """ Get root output directory for each run """
    return os.path.join(args.checkpoint_dir, run_name)

def cal_dice2(pred, target):
    # for 2 channels
    N = pred.size(0)
    C = 2
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred.unsqueeze(1), 1.)

    intersection = pred_mask.cpu() * target_mask
    summ = pred_mask.cpu() + target_mask

    intersection = intersection.sum(0).type(torch.float64)
    summ = summ.sum(0).type(torch.float64)

    eps = torch.rand(C, dtype=torch.float64)
    eps = eps.fill_(0.0000001)
    summ += eps

    dice = 2 * intersection / summ

    return dice, intersection, summ
    

def get_d4x_label(labels):
    labels = labels.type(torch.FloatTensor)
    label_d4x = F.max_pool3d(labels, kernel_size=(2,4,4), stride=(2,4,4))
    label_d4x = label_d4x.type(torch.LongTensor)
    return label_d4x

def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def vis_result(data, scores, labels, out_dir):
    """
        Args:
            data: (N, C, D, H, W)
            scores: (N, C, D, H, W)
            labels: (N, D, H, W)
    """
    COLOR_MAP1 = [(0, 0, 0), (220, 20, 60), (0, 0, 142)]
    ndata = data.clone().cpu()  # [1,1,98,240,240]
    nscores = scores.clone().cpu()  # [1,19,98,240,240]
    nlabels = labels.clone().cpu()  # [1,98,240,240]
    n_class = nscores.size(1)

    pred = F.softmax(nscores, dim=1)
    _, pred_label = torch.max(pred, dim=1)

    npdata = ndata.squeeze().numpy()
    nppred_label = pred_label.squeeze().numpy().astype(np.int32)
    nplabels = nlabels.squeeze().numpy().astype(np.int32)

    # scale data to CT intensity
    npdata = npdata * 1000 - 100

    npdata = np.tile(npdata, (1, 1, 3))
    tmp = np.zeros(nppred_label.shape).astype(np.int32)
    nppred = np.concatenate((tmp, nppred_label, nplabels), axis=2)

    itk_data = sitk.GetImageFromArray(npdata)
    itk_pred = sitk.GetImageFromArray(nppred)
    #itk_lebel = sitk.GetImageFromArray(nplabels)

    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    sitk.WriteImage(itk_data, osp.join(out_dir, 'data.nii.gz'))
    sitk.WriteImage(itk_pred, osp.join(out_dir, 'label.nii.gz'))
    #sitk.WriteImage(itk_lebel, osp.join(out_dir, 'gt.nii.gz'))
    #pdb.set_trace()

def vis_result_contour(data, scores, labels, out_dir):
    """
        Args:
            data: (N, C, D, H, W)
            scores: (N, C, D, H, W)
            labels: (N, D, H, W)
    """
    COLOR_MAP1 = [(0, 0, 0), (220, 20, 60), (0, 0, 142)]
    ndata = data.clone().cpu()  # [1,1,98,240,240]
    nscores = scores.clone().cpu()  # [1,19,98,240,240]
    nlabels = labels.clone().cpu()  # [1,98,240,240]
    n_class = nscores.size(1)

    pred = F.softmax(nscores, dim=1)
    _, pred_label = torch.max(pred, dim=1)

    npdata = ndata.squeeze().numpy()
    nppred_label = pred_label.squeeze().numpy().astype(np.int32) # (D, H, W)
    nplabels = nlabels.squeeze().numpy().astype(np.uint8)
    nplabels[nplabels == 2] = 0

    #pdb.set_trace()
    D = nppred_label.shape[0]
    nppred_contour = np.zeros(nppred_label.shape)
    nppred_label = nppred_label.astype(np.uint8)
    for i in range(D):
        _, contours, hierarchy = cv2.findContours(nppred_label[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
        #if len(contours) > 0:
        for j in range(len(contours)):
            contour = contours[j]
            contour = contour[:, 0 , :]
            for k in range(len(contour)):
                nppred_contour[i, contour[k][1], contour[k][0]] = 1

        _, contours_gt, hierarchy = cv2.findContours(nplabels[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
        for j in range(len(contours_gt)):
            contour_gt = contours_gt[j]
            contour_gt = contour_gt[:, 0 , :]
            for k in range(len(contour_gt)):
                nppred_contour[i, contour_gt[k][1], contour_gt[k][0]] = 2
    
    #pdb.set_trace()


    # scale data to CT intensity
    npdata = npdata * 1000 - 100

    npdata = np.tile(npdata, (1, 1, 2))
    tmp = np.zeros(nppred_label.shape).astype(np.int32)
    nppred = np.concatenate((tmp, nppred_contour), axis=2)

    itk_data = sitk.GetImageFromArray(npdata)
    itk_pred = sitk.GetImageFromArray(nppred)
    #itk_lebel = sitk.GetImageFromArray(nplabels)

    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    sitk.WriteImage(itk_data, osp.join(out_dir, 'data.nii.gz'))
    sitk.WriteImage(itk_pred, osp.join(out_dir, 'label.nii.gz'))
    #sitk.WriteImage(itk_lebel, osp.join(out_dir, 'gt.nii.gz'))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
