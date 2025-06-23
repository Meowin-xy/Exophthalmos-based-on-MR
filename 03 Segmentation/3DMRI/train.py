"""
@author: Xinyi Gou
Time: 2024-12-23
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

from models import transforms
from models.datasetEyeball import datasetEyeball
from models import UNet

from models.LRScheduler import LRScheduler
from models.metrics import dice
from models import losses
from models.util import *

from tensorboardX import SummaryWriter
torch.cuda.set_device(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args):
    run_name = get_run_name()
    date = run_name.split('_')[0]

    save_result_folder = os.path.join('E:/gxy2025/tao-cal/Output', args.model, f'{args.model}_{date[:11]}')
    
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

    # make logger file
    code_folder = os.path.join(save_result_folder, 'code')
    if os.path.exists(code_folder):
        shutil.rmtree(code_folder)
    project_root = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
    shutil.copytree(project_root, code_folder, ...)
    logging.basicConfig(filename=os.path.join(save_result_folder, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    writer = SummaryWriter(os.path.join(save_result_folder, 'log'))

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # train and val data transforms:
    train_transforms = transforms.Compose([
        # transforms.ExtractPatch((144, 144, 144), p_tumor=0.5),
        transforms.RandomRotation(p=0.5, angle_range=[0, 15]),
        transforms.Mirroring(p=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        # transforms.ExtractPatch((144, 144, 144), p_tumor=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])
    all_paths = []
    Image3D_path = os.path.join(args.trainFolder, 'images')
    print(Image3D_path)
    for item in os.listdir(Image3D_path):
        f_path = os.path.join(Image3D_path, item)
        all_paths.append(f_path)

    print("All data number: ", len(all_paths))
    # ***************** 五折交叉验证 *******************
    folder = KFold(n_splits=5, random_state=30, shuffle=True)
    train_paths = []  # 存放5折的训练集划分
    val_paths = []  # 存放5折的验证集划分
    for k, (Trindex, Tsindex) in enumerate(folder.split(all_paths)):
        train_paths.append(np.array(all_paths)[Trindex].tolist())
        val_paths.append(np.array(all_paths)[Tsindex].tolist())
    df = pd.DataFrame(data=train_paths, index=['0', '1', '2', '3', '4'])
    df.to_csv(os.path.join(save_result_folder, 'train.csv'))
    df1 = pd.DataFrame(data=val_paths, index=['0', '1', '2', '3', '4'])
    df1.to_csv(os.path.join(save_result_folder, 'val.csv'))

    train_set = datasetEyeball(train_paths[0], transform=train_transforms)
    val_set = datasetEyeball(val_paths[0], transform=val_transforms)

    # dataloader:
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    if args.model == 'Unet':
        model = UNet.BaselineUNet(1, 2, 16)
        print("use baselineUNet")
    else:
        print("no model !!! ")

    criterion = losses.Dice_and_FocalLoss()
    print("use Dice_and_FocalLoss")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001, betas=(0.9, 0.99))
    
    all_niters = len(train_loader)
    lr_scheduler = LRScheduler(optimizer, all_niters, args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    best_dice = 0.0
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        trainloader_iter = iter(train_loader)
        model.train()
        for i in range(all_niters):
            sample = next(trainloader_iter)
            lr_scheduler.update(i, epoch)
      
            img, label = sample['input'], sample['target']
            inputs = img.to(device)  # 1 4 128 160 160
            labels = label.to(device)  # 1 128 160 160
            optimizer.zero_grad() 
            result = model(inputs)
            loss = criterion(result, labels)

            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()

            logging.info('Epoch: {}/{}, step: {}/{}, batch_loss: {}'.format(epoch, args.epochs, i, all_niters, loss.item()))

            writer.add_scalar(
                'loss', loss.item(), epoch * all_niters + i)

        # ************* val *************
        model.eval()
        print('val model on validation set...')
        Dice = 0.0
        
        with torch.no_grad():
            val_iter = iter(val_loader)
            for i in range(len(val_iter)):
                sample = next(val_iter)
                lr_scheduler.update(i, epoch)
                img, label = sample['input'], sample['target']
                inputs = img.to(device)  # 1 4 128 160 160
                labels = label.to(device)  # 1 128 160 160
                result = model(inputs)
                dice_score = dice(result.detach(), labels.detach()).item()
                print("dice: ", dice_score)

                Dice += dice_score
            epoch_dice = Dice / len(val_loader)
            print("epoch Val Dice: ", epoch_dice)
            logging.info('epoch Val Dice: {}'.format(epoch_dice))
            writer.add_scalar('val_dice', epoch_dice, epoch + 1)

            # 保存表现最好的模型
            save_model_path = os.path.join(save_result_folder, 'save_models')
            if not os.path.exists(save_model_path):
                os.mkdir(save_model_path)
            if best_dice < epoch_dice:
                best_dice = epoch_dice
                model_save_name = f'model_epoch_best.pth'
                torch.save(model.state_dict(), os.path.join(save_model_path, model_save_name))
                print(f"Model saved as {model_save_name}")
            logging.info("best test acc: {}".format(best_dice))

            # Save model every 50 epochs
            if (epoch + 1) % 50 == 0:
                model_save_name = f'model_epoch_{epoch + 1}.pth'
                torch.save(model.state_dict(), os.path.join(save_model_path, model_save_name))
                print(f"Model saved as {model_save_name}")
            if (epoch + 1) == args.epochs:
                model_save_name = f'model_epoch_last.pth'
                torch.save(model.state_dict(), os.path.join(save_model_path, model_save_name))
                print(f"Model saved as {model_save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("--trainFolder", type=str, default='E:/data/all/train', help="path to the config file")  # 添加参数
    parser.add_argument("--testFolder", type=str, default='E:/data/all/validation', help="path to the config file")  # 添加参数
    parser.add_argument("--train_batch_size", type=int, default=2, help="path to the config file")
    parser.add_argument("--val_batch_size", type=int, default=2, help="path to the config file")
    parser.add_argument("--test_batch_size", type=int, default=2, help="path to the config file")
    parser.add_argument("--lr", type=int, default=1e-4, help="path to the config file")
    parser.add_argument('--lr_mode', default='poly', type=str, help='lr scheluder  step | poly | cos | linear')
    parser.add_argument('--step', default='50, 100, 150', type=str, help='lr scheluder step')
    parser.add_argument('--decay_factor', default=0.5, type=str, help='lr scheluder step decay_factor')
    parser.add_argument('--warmup_mode', default='linear', type=str, help='warmup_mode')
    parser.add_argument('--warmup_lr', default=0.00001, type=float, help='warmup_lr')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='warmup_epochs')
    
    parser.add_argument("--model", type=str, default='Unet', help="Unet ")
    parser.add_argument("--epochs", type=int, default=300, help="path to the config file")
    parser.add_argument("--gpu", type=int, default=0, help="path to the config file")

    args = parser.parse_args()
    main(args)
