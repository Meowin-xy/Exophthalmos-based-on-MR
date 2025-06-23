"""
@author: zhengyong Huang
Time: 2024-12-02 
"""
import os
import numpy as np
import sys
import argparse
import yaml
import pathlib
import logging, shutil
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import torch

from utils.util import *
from utils.metrics import dice
from dataset import transforms
from utils import losses
from dataset.datasetEyeball import datasetEyeball

from utils.LRScheduler import LRScheduler

from models import UNet
from tensorboardX import SummaryWriter

torch.cuda.set_device(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def save_nii(id, data, label, output, save_path):
    save_nii_path = os.path.join(save_path, 'pred_result', id)
    if not os.path.exists(save_nii_path):
        os.makedirs(save_nii_path)

    output = (output > 0.5).astype(np.int16)
    data_path = os.path.join(r'G:\finaldata0528\2DMRI\Eyeball\2506_2Dcrop\images', id+'.nii.gz')
    img = sitk.ReadImage(data_path)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    # print(data.shape)
    # print(label.shape)

    data_nii = sitk.GetImageFromArray(data[0,0,...])
    label_nii = sitk.GetImageFromArray(label[0,0,...])
    output_nii = sitk.GetImageFromArray(output[0,0,...])

    data_nii.SetOrigin(origin)
    data_nii.SetSpacing(spacing)
    data_nii.SetDirection(direction)
    sitk.WriteImage(data_nii, save_nii_path+'/image.nii.gz')

    label_nii.SetOrigin(origin)
    label_nii.SetSpacing(spacing)
    label_nii.SetDirection(direction)
    sitk.WriteImage(label_nii, save_nii_path+'/label.nii.gz')
    
    output_nii.SetOrigin(origin)
    output_nii.SetSpacing(spacing)
    output_nii.SetDirection(direction)
    sitk.WriteImage(output_nii, save_nii_path+'/pred.nii.gz')


def main(args):
    save_result_folder = os.path.join(args.model_path.split("save_models")[0], "test_result")
    
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

    logging.basicConfig(filename=save_result_folder + "/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

 

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    
    test_transforms = transforms.Compose([
        # transforms.ExtractPatch((144, 144, 144), p_tumor=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])
    all_paths = []
    Image3D_path = os.path.join(args.testFolder, 'images')
    for item in os.listdir(Image3D_path):
        f_path = os.path.join(Image3D_path, item)
        all_paths.append(f_path)

    print("All data number: ", len(all_paths))


    test_set = datasetEyeball(all_paths, transform=test_transforms)
  

    # dataloader:
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # 查询参数量
    # c = models.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels,  n_cls, n_filters, reduction=2)
    # print(sum(p.numel() for p in c.parameters()))

    if args.model=='Unet':
        # model = models.BaselineUNet_1(in_channels, n_cls, n_filters, init_type='xavier')
        model = UNet.BaselineUNet(1, 2, 16)
        # model = models.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels, n_cls, n_filters)

    else:
        model = SEnet.FastSmoothSENormDeepUNet_supervision_skip_no_drop(in_channels=1, n_cls=2, n_filters=16, reduction=4)

    model.load_state_dict(torch.load(args.model_path))

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    best_dice = 0.0
    model.eval()
    print('val model on test set...')
    Dice = 0.0
    with torch.no_grad():
        test_iter = iter(test_loader)
        for i in range(len(test_iter)):
            sample = next(test_iter)
            img, label, id= sample['input'], sample['target'], sample['id']
            inputs = img.to(device)  # 1 4 128 160 160
            labels = label.to(device)  # 1 128 160 160
            result = model(inputs)
            dice_score = dice(result.detach(), labels.detach()).item()
            logging.info('ID: {}, Dice: {}'.format(id[0], dice_score))
            Dice += dice_score
            save_nii(id[0], img.detach().cpu().numpy(), label.detach().cpu().numpy(), result.detach().cpu().numpy(), save_result_folder)

        avg_dice = Dice/len(test_loader)
        logging.info('Avg test Dice: {}'.format(avg_dice))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("--model_path", type=str, default=r'G:\finaldata0528\Code\Segment\2DEyeball\save_models\model_epoch_best.pth', help="path to the config file")  # 添加参数
    parser.add_argument("--testFolder", type=str, default=r'G:\finaldata0528\2DMRI\Eyeball\2506_2Dcrop', help="path to the config file")  # 添加参数
    parser.add_argument("--model", type=str, default='Unet', help="Unet")
    parser.add_argument("--gpu", type=int, default=0, help="path to the config file")

    args = parser.parse_args()
    main(args)
