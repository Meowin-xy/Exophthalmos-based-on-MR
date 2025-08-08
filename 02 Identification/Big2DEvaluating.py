import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy import ndimage  # 添加导入，用于连通域分析

# 数据路径设置
split_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2Dsplit_data")
val_img_dir = os.path.join(split_data_dir, "validation", "images")
val_label_dir = os.path.join(split_data_dir, "validation", "labels")

# 创建模型保存目录
model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(model_save_dir, exist_ok=True)

# 自定义数据集类
class ZygomaticDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and not f.startswith('._')])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx])
        
        # 加载NIfTI文件
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        
        # 获取数据
        img_data = img_nii.get_fdata()
        label_data = label_nii.get_fdata()
        
        # 只取第一个切片(维度是320x112x1)
        img_slice = img_data[:, :, 0]
        label_slice = label_data[:, :, 0]
        
        # 将标签二值化 - 颧突位置为1，其余为0
        binary_mask = np.zeros_like(label_slice)
        if np.max(label_slice) > 0:  # 确保有标签
            binary_mask = (label_slice > 0).astype(np.float32)
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0)  # 添加通道维度
        mask_tensor = torch.from_numpy(binary_mask).float().unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor, mask_tensor

# 定义U-Net模型的各个模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 动态padding以匹配尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

# 定义完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.sigmoid(x)

# 准备数据加载器
def get_dataloader(batch_size=4):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建验证数据集
    val_dataset = ZygomaticDataset(val_img_dir, val_label_dir, transform=transform)
    
    # 创建数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return val_loader

# 可视化预测结果
def visualize_predictions(model, val_loader, num_samples=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 获取一批数据
    dataiter = iter(val_loader)
    images, masks = next(dataiter)
    
    with torch.no_grad():
        images = images.to(device)
        predictions = model(images)
        
        # 修改预测结果处理方式 - 使用大于0.5的区域重心
        modified_predictions = []
        thresholded_regions = []  # 存储大于0.5的区域
        
        for pred in predictions:
            pred_np = pred[0].cpu().numpy()  # 转为numpy数组处理
            height = pred_np.shape[0]
            half_height = height // 2
            
            # 将图像分为上下两半区
            upper_half = pred_np[:half_height, :]
            lower_half = pred_np[half_height:, :]
            
            # 创建空白掩码
            result_mask = np.zeros_like(pred_np)
            
            # 创建用于存储大于0.5区域的掩码
            threshold_mask = np.zeros_like(pred_np)
            
            # 处理上半区域 - 找到大于0.5的区域并计算重心
            upper_threshold = upper_half > 0.5
            if np.any(upper_threshold):  # 确保有大于0.5的区域
                # 进行连通域分析
                labeled_upper, num_features_upper = ndimage.label(upper_threshold)
                
                if num_features_upper > 0:
                    # 计算各连通域的大小
                    sizes_upper = ndimage.sum(upper_threshold, labeled_upper, range(1, num_features_upper + 1))
                    
                    # 找出最大连通域的标签
                    max_label_upper = np.argmax(sizes_upper) + 1
                    
                    # 提取最大连通域
                    max_region_upper = (labeled_upper == max_label_upper)
                    
                    # 在阈值掩码中标记上半区最大连通域
                    threshold_mask[:half_height, :][max_region_upper] = 1
                    
                    # 计算最大连通域的重心
                    upper_y, upper_x = np.where(max_region_upper)
                    if len(upper_y) > 0:  # 确保有点存在
                        upper_center_y = int(np.mean(upper_y))
                        upper_center_x = int(np.mean(upper_x))
                        # 在结果掩码中标记重心点
                        result_mask[upper_center_y, upper_center_x] = 1
            
            # 处理下半区域 - 找到大于0.5的区域并计算重心
            lower_threshold = lower_half > 0.5
            if np.any(lower_threshold):  # 确保有大于0.5的区域
                # 进行连通域分析
                labeled_lower, num_features_lower = ndimage.label(lower_threshold)
                
                if num_features_lower > 0:
                    # 计算各连通域的大小
                    sizes_lower = ndimage.sum(lower_threshold, labeled_lower, range(1, num_features_lower + 1))
                    
                    # 找出最大连通域的标签
                    max_label_lower = np.argmax(sizes_lower) + 1
                    
                    # 提取最大连通域
                    max_region_lower = (labeled_lower == max_label_lower)
                    
                    # 在阈值掩码中标记下半区最大连通域，注意调整索引
                    threshold_mask[half_height:, :][max_region_lower] = 1
                    
                    # 计算最大连通域的重心，注意调整y坐标
                    lower_y, lower_x = np.where(max_region_lower)
                    if len(lower_y) > 0:  # 确保有点存在
                        lower_center_y = int(np.mean(lower_y)) + half_height
                        lower_center_x = int(np.mean(lower_x))
                        # 在结果掩码中标记重心点
                        result_mask[lower_center_y, lower_center_x] = 1
            
            modified_predictions.append(result_mask)
            thresholded_regions.append(threshold_mask)
        
        # 转换为正确的形状以便显示
        predictions = np.array(modified_predictions)[:, np.newaxis, :, :]
        thresholded_regions = np.array(thresholded_regions)[:, np.newaxis, :, :]
    
    # 转回CPU进行可视化
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    # 显示结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # 原始图像
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        
        # 只标记Ground Truth中值为1的点
        gt_y, gt_x = np.where(masks[i, 0] == 1)
        if len(gt_y) > 0:
            axes[i, 0].scatter(gt_x, gt_y, c='r', marker='o', s=15, alpha=0.7)
            
        axes[i, 0].set_title('Image + Ground Truth')
        axes[i, 0].axis('off')
        
        # 原始图像 + 大于0.5的区域
        axes[i, 1].imshow(images[i, 0], cmap='gray')
        
        # 添加大于0.5的区域作为半透明的蓝色区域
        threshold_mask = thresholded_regions[i, 0]
        # 创建一个RGBA颜色图像，其中非零部分为蓝色，半透明
        color_mask = np.zeros((*threshold_mask.shape, 4), dtype=np.float32)
        color_mask[threshold_mask > 0] = [0.0, 0.0, 1.0, 0.3]  # 蓝色，透明度0.3
        axes[i, 1].imshow(color_mask)
        
        axes[i, 1].set_title('Image + Largest Connected Components')
        axes[i, 1].axis('off')
        
        # 原始图像 + 重心点
        axes[i, 2].imshow(images[i, 0], cmap='gray')
        
        # 添加大于0.5的区域作为半透明的蓝色区域
        axes[i, 2].imshow(color_mask)
        
        # 只标记Prediction中值为1的点（即重心点）
        pred_y, pred_x = np.where(predictions[i, 0] == 1)
        if len(pred_y) > 0:
            axes[i, 2].scatter(pred_x, pred_y, c='r', marker='x', s=30, alpha=1.0)
            
        axes[i, 2].set_title('Image + Connected Components + Centroids')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 计算预测点和真实标签点之间的平均距离
def calculate_average_distances(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_upper_distance = 0
    total_lower_distance = 0
    valid_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="计算距离")):
            images = images.to(device)
            masks = masks.cpu().numpy()
            predictions = model(images)
            
            # 将预测结果处理为只有两个点
            batch_size = images.shape[0]
            for i in range(batch_size):
                # 获取文件名（用于输出）
                sample_idx = batch_idx * batch_size + i
                file_name = val_loader.dataset.img_files[sample_idx] if sample_idx < len(val_loader.dataset.img_files) else f"Sample_{sample_idx}"
                
                # 处理预测结果 - 使用区域重心方法
                pred_np = predictions[i, 0].cpu().numpy()
                height = pred_np.shape[0]
                half_height = height // 2
                
                # 将图像分为上下两半区
                upper_half = pred_np[:half_height, :]
                lower_half = pred_np[half_height:, :]
                
                # 创建空白掩码
                result_mask = np.zeros_like(pred_np)
                
                # 处理上半区域 - 找到大于0.5的区域并计算重心
                upper_pred_y, upper_pred_x = None, None
                upper_threshold = upper_half > 0.5
                if np.any(upper_threshold):  # 确保有大于0.5的区域
                    # 进行连通域分析
                    labeled_upper, num_features_upper = ndimage.label(upper_threshold)
                    
                    if num_features_upper > 0:
                        # 计算各连通域的大小
                        sizes_upper = ndimage.sum(upper_threshold, labeled_upper, range(1, num_features_upper + 1))
                        
                        # 找出最大连通域的标签
                        max_label_upper = np.argmax(sizes_upper) + 1
                        
                        # 提取最大连通域
                        max_region_upper = (labeled_upper == max_label_upper)
                        
                        # 计算最大连通域的重心
                        upper_y, upper_x = np.where(max_region_upper)
                        if len(upper_y) > 0:  # 确保有点存在
                            upper_pred_y = int(np.mean(upper_y))
                            upper_pred_x = int(np.mean(upper_x))
                            # 在结果掩码中标记重心点
                            result_mask[upper_pred_y, upper_pred_x] = 1
                
                # 处理下半区域 - 找到大于0.5的区域并计算重心
                lower_pred_y, lower_pred_x = None, None
                lower_threshold = lower_half > 0.5
                if np.any(lower_threshold):  # 确保有大于0.5的区域
                    # 进行连通域分析
                    labeled_lower, num_features_lower = ndimage.label(lower_threshold)
                    
                    if num_features_lower > 0:
                        # 计算各连通域的大小
                        sizes_lower = ndimage.sum(lower_threshold, labeled_lower, range(1, num_features_lower + 1))
                        
                        # 找出最大连通域的标签
                        max_label_lower = np.argmax(sizes_lower) + 1
                        
                        # 提取最大连通域
                        max_region_lower = (labeled_lower == max_label_lower)
                        
                        # 计算最大连通域的重心，注意调整y坐标
                        lower_y, lower_x = np.where(max_region_lower)
                        if len(lower_y) > 0:  # 确保有点存在
                            lower_pred_y = int(np.mean(lower_y)) + half_height
                            lower_pred_x = int(np.mean(lower_x))
                            # 在结果掩码中标记重心点
                            result_mask[lower_pred_y, lower_pred_x] = 1
                
                # 处理真实标签
                mask_np = masks[i, 0]
                
                # 找到真实标签中值为1的点
                gt_y, gt_x = np.where(mask_np == 1)
                
                # 分离上下半区的真实标签点
                upper_mask_y = gt_y[gt_y < half_height] if len(gt_y) > 0 else []
                upper_mask_x = gt_x[gt_y < half_height] if len(gt_y) > 0 else []
                
                lower_mask_y = gt_y[gt_y >= half_height] if len(gt_y) > 0 else []
                lower_mask_x = gt_x[gt_y >= half_height] if len(gt_y) > 0 else []
                
                print(f"\n样本 {file_name}:")
                
                # 输出上半区点位置和距离
                if upper_pred_y is not None and len(upper_mask_y) > 0:
                    upper_pred_point = np.array([upper_pred_y, upper_pred_x])
                    upper_mask_point = np.array([upper_mask_y[0], upper_mask_x[0]])
                    upper_distance = np.sqrt(np.sum((upper_pred_point - upper_mask_point)**2))
                    
                    print(f"  上半区 - 预测点: ({upper_pred_y}, {upper_pred_x}), 真实点: ({upper_mask_y[0]}, {upper_mask_x[0]}), 距离: {upper_distance:.2f}像素")
                else:
                    upper_distance = None
                    print("  上半区 - 没有找到有效的预测点或真实标签点")
                
                # 输出下半区点位置和距离
                if lower_pred_y is not None and len(lower_mask_y) > 0:
                    lower_pred_point = np.array([lower_pred_y, lower_pred_x])
                    lower_mask_point = np.array([lower_mask_y[0], lower_mask_x[0]])
                    lower_distance = np.sqrt(np.sum((lower_pred_point - lower_mask_point)**2))
                    
                    print(f"  下半区 - 预测点: ({lower_pred_y}, {lower_pred_x}), 真实点: ({lower_mask_y[0]}, {lower_mask_x[0]}), 距离: {lower_distance:.2f}像素")
                else:
                    lower_distance = None
                    print("  下半区 - 没有找到有效的预测点或真实标签点")
                
                # 只有当上下半区都有有效点时才计入统计
                if upper_distance is not None and lower_distance is not None:
                    total_upper_distance += upper_distance
                    total_lower_distance += lower_distance
                    valid_samples += 1
    
    print("\n===== 统计信息 =====")
    if valid_samples > 0:
        avg_upper_distance = total_upper_distance / valid_samples
        avg_lower_distance = total_lower_distance / valid_samples
        avg_total_distance = (total_upper_distance + total_lower_distance) / (valid_samples * 2)
        
        print(f"上半区平均距离: {avg_upper_distance:.2f} 像素")
        print(f"下半区平均距离: {avg_lower_distance:.2f} 像素")
        print(f"总平均距离: {avg_total_distance:.2f} 像素")
        print(f"有效样本数: {valid_samples}")
        
        return avg_upper_distance, avg_lower_distance, avg_total_distance
    else:
        print("没有找到有效样本进行距离计算")
        return None, None, None

# 可视化距离大于阈值的样本
def visualize_large_distance_samples(model, val_loader, distance_threshold=5.0, prob_threshold=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    large_distance_samples = []  # 存储距离大于阈值的样本信息
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Screening samples with large distance")):
            images = images.to(device)
            masks_np = masks.cpu().numpy()
            predictions = model(images)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                file_name = val_loader.dataset.img_files[sample_idx] if sample_idx < len(val_loader.dataset.img_files) else f"Sample_{sample_idx}"
                
                # 处理预测结果 - 使用区域重心方法
                pred_np = predictions[i, 0].cpu().numpy()
                img_np = images[i, 0].cpu().numpy()
                mask_np = masks_np[i, 0]
                
                height = pred_np.shape[0]
                half_height = height // 2
                
                # 将图像分为上下两半区
                upper_half = pred_np[:half_height, :]
                lower_half = pred_np[half_height:, :]
                
                # 创建空白掩码和阈值掩码
                result_mask = np.zeros_like(pred_np)
                threshold_mask = np.zeros_like(pred_np)
                prob_map = np.copy(pred_np)  # 保存完整的概率图
                
                # 处理上半区域 - 找到大于阈值的区域并计算重心
                upper_pred_y, upper_pred_x = None, None
                upper_threshold = upper_half > prob_threshold
                if np.any(upper_threshold):  # 确保有大于阈值的区域
                    # 进行连通域分析
                    labeled_upper, num_features_upper = ndimage.label(upper_threshold)
                    
                    if num_features_upper > 0:
                        # 计算各连通域的大小
                        sizes_upper = ndimage.sum(upper_threshold, labeled_upper, range(1, num_features_upper + 1))
                        
                        # 找出最大连通域的标签
                        max_label_upper = np.argmax(sizes_upper) + 1
                        
                        # 提取最大连通域
                        max_region_upper = (labeled_upper == max_label_upper)
                        
                        # 在阈值掩码中标记上半区最大连通域
                        threshold_mask[:half_height, :][max_region_upper] = 1
                        
                        # 计算最大连通域的重心
                        upper_y, upper_x = np.where(max_region_upper)
                        if len(upper_y) > 0:  # 确保有点存在
                            upper_pred_y = int(np.mean(upper_y))
                            upper_pred_x = int(np.mean(upper_x))
                            # 在结果掩码中标记重心点
                            result_mask[upper_pred_y, upper_pred_x] = 1
                
                # 处理下半区域 - 找到大于阈值的区域并计算重心
                lower_pred_y, lower_pred_x = None, None
                lower_threshold = lower_half > prob_threshold
                if np.any(lower_threshold):  # 确保有大于阈值的区域
                    # 进行连通域分析
                    labeled_lower, num_features_lower = ndimage.label(lower_threshold)
                    
                    if num_features_lower > 0:
                        # 计算各连通域的大小
                        sizes_lower = ndimage.sum(lower_threshold, labeled_lower, range(1, num_features_lower + 1))
                        
                        # 找出最大连通域的标签
                        max_label_lower = np.argmax(sizes_lower) + 1
                        
                        # 提取最大连通域
                        max_region_lower = (labeled_lower == max_label_lower)
                        
                        # 在阈值掩码中标记下半区最大连通域
                        threshold_mask[half_height:, :][max_region_lower] = 1
                        
                        # 计算最大连通域的重心，注意调整y坐标
                        lower_y, lower_x = np.where(max_region_lower)
                        if len(lower_y) > 0:  # 确保有点存在
                            lower_pred_y = int(np.mean(lower_y)) + half_height
                            lower_pred_x = int(np.mean(lower_x))
                            # 在结果掩码中标记重心点
                            result_mask[lower_pred_y, lower_pred_x] = 1
                
                # 找到真实标签中值为1的点
                gt_y, gt_x = np.where(mask_np == 1)
                
                # 分离上下半区的真实标签点
                upper_mask_y = gt_y[gt_y < half_height] if len(gt_y) > 0 else []
                upper_mask_x = gt_x[gt_y < half_height] if len(gt_y) > 0 else []
                
                lower_mask_y = gt_y[gt_y >= half_height] if len(gt_y) > 0 else []
                lower_mask_x = gt_x[gt_y >= half_height] if len(gt_y) > 0 else []
                
                # 计算距离
                upper_distance = None
                lower_distance = None
                
                if upper_pred_y is not None and len(upper_mask_y) > 0:
                    upper_pred_point = np.array([upper_pred_y, upper_pred_x])
                    upper_mask_point = np.array([upper_mask_y[0], upper_mask_x[0]])
                    upper_distance = np.sqrt(np.sum((upper_pred_point - upper_mask_point)**2))
                
                if lower_pred_y is not None and len(lower_mask_y) > 0:
                    lower_pred_point = np.array([lower_pred_y, lower_pred_x])
                    lower_mask_point = np.array([lower_mask_y[0], lower_mask_x[0]])
                    lower_distance = np.sqrt(np.sum((lower_pred_point - lower_mask_point)**2))
                
                # 如果任一距离大于阈值，则保存样本信息
                if (upper_distance is not None and upper_distance > distance_threshold) or \
                   (lower_distance is not None and lower_distance > distance_threshold):
                    sample_info = {
                        'file_name': file_name,
                        'image': img_np,
                        'mask': mask_np,
                        'prediction': result_mask,
                        'threshold_mask': threshold_mask,  # 添加阈值掩码
                        'prob_map': prob_map,  # 添加完整概率图
                        'upper_pred': (upper_pred_y, upper_pred_x) if upper_pred_y is not None else None,
                        'lower_pred': (lower_pred_y, lower_pred_x) if lower_pred_y is not None else None,
                        'upper_gt': (upper_mask_y[0], upper_mask_x[0]) if len(upper_mask_y) > 0 else None,
                        'lower_gt': (lower_mask_y[0], lower_mask_x[0]) if len(lower_mask_y) > 0 else None,
                        'upper_distance': upper_distance,
                        'lower_distance': lower_distance
                    }
                    large_distance_samples.append(sample_info)
    
    # 可视化距离大的样本
    if len(large_distance_samples) > 0:
        print(f"\n找到 {len(large_distance_samples)} 个距离大于 {distance_threshold} 像素的样本")
        print(f"使用预测概率阈值: {prob_threshold}")
        
        # 专门打印所有距离大于阈值的样本文件名
        print("\n距离大于阈值的样本文件名列表:")
        for i, sample in enumerate(large_distance_samples):
            upper_dist = sample['upper_distance'] if sample['upper_distance'] else 0
            lower_dist = sample['lower_distance'] if sample['lower_distance'] else 0
            max_dist = max(upper_dist or 0, lower_dist or 0)  # 使用较大的距离
            print(f"{i+1}. {sample['file_name']} - 最大距离: {max_dist:.2f}像素")
        
        print("\n请关闭当前图像窗口查看下一个样本...")
        
        for i, sample in enumerate(large_distance_samples):
            # 为每个样本创建一个新的图形，包含两个子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 第一个子图：显示原始图像和阈值区域
            axes[0].imshow(sample['image'], cmap='gray')
            
            # 创建阈值区域的彩色蒙版
            threshold_mask = sample['threshold_mask']
            color_mask = np.zeros((*threshold_mask.shape, 4), dtype=np.float32)
            color_mask[threshold_mask > 0] = [0.0, 0.0, 1.0, 0.3]  # 蓝色，透明度0.3
            axes[0].imshow(color_mask)
            
            # 标记真实标签点
            if sample['upper_gt']:
                y, x = sample['upper_gt']
                axes[0].scatter(x, y, c='r', marker='o', s=30, alpha=0.7, label='Upper Ground Truth')
            
            if sample['lower_gt']:
                y, x = sample['lower_gt']
                axes[0].scatter(x, y, c='r', marker='o', s=30, alpha=0.7, label='Lower Ground Truth')
            
            axes[0].set_title(f'Image + Largest Connected Regions (>{prob_threshold}) + Ground Truth')
            axes[0].legend(loc='upper right')
            axes[0].axis('off')
            
            # 第二个子图：显示原始概率热图
            im = axes[1].imshow(sample['prob_map'], cmap='jet', vmin=0, vmax=1)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 标记真实标签点
            if sample['upper_gt']:
                y, x = sample['upper_gt']
                axes[1].scatter(x, y, c='w', marker='o', s=30, alpha=1.0, label='Ground Truth')
            
            if sample['lower_gt']:
                y, x = sample['lower_gt']
                axes[1].scatter(x, y, c='w', marker='o', s=30, alpha=1.0)
            
            # 添加阈值线
            axes[1].set_title(f'Probability Map (Threshold: {prob_threshold})')
            axes[1].axis('off')
            
            # 第三个子图：显示原始图像、阈值区域、重心点和连接线
            axes[2].imshow(sample['image'], cmap='gray')
            axes[2].imshow(color_mask)
            
            # 标记真实标签点
            if sample['upper_gt']:
                y, x = sample['upper_gt']
                axes[2].scatter(x, y, c='r', marker='o', s=30, alpha=0.7, label='Upper Ground Truth')
            
            if sample['lower_gt']:
                y, x = sample['lower_gt']
                axes[2].scatter(x, y, c='r', marker='o', s=30, alpha=0.7, label='Lower Ground Truth')
            
            # 标记预测点
            if sample['upper_pred']:
                y, x = sample['upper_pred']
                upper_dist = sample['upper_distance']
                axes[2].scatter(x, y, c='b', marker='x', s=30, alpha=0.7, label=f'Upper Prediction (Distance: {upper_dist:.2f})')
            
            if sample['lower_pred']:
                y, x = sample['lower_pred']
                lower_dist = sample['lower_distance']
                axes[2].scatter(x, y, c='g', marker='x', s=30, alpha=0.7, label=f'Lower Prediction (Distance: {lower_dist:.2f})')
            
            # 添加距离线
            if sample['upper_gt'] and sample['upper_pred']:
                gy, gx = sample['upper_gt']
                py, px = sample['upper_pred']
                axes[2].plot([gx, px], [gy, py], 'b--', linewidth=1, alpha=0.5)
            
            if sample['lower_gt'] and sample['lower_pred']:
                gy, gx = sample['lower_gt']
                py, px = sample['lower_pred']
                axes[2].plot([gx, px], [gy, py], 'g--', linewidth=1, alpha=0.5)
            
            axes[2].set_title('Image + Connected Components + Centroids')
            axes[2].legend(loc='upper right')
            axes[2].axis('off')
            
            plt.suptitle(f"Sample {i+1}/{len(large_distance_samples)}: {sample['file_name']}", fontsize=14)
            plt.tight_layout()
            
            # 显示当前样本图像并等待用户关闭
            plt.show()
    else:
        print(f"\n没有找到距离大于 {distance_threshold} 像素的样本")
def visualize_all_samples(model, val_loader, prob_threshold=0.5):
    """
    逐一可视化验证集中的所有样本
    —— 显示：原始图 + 阈值区域 + GT点、概率热图、原始图+预测点三联图
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage      # 函数内部确保依赖
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Visualizing all samples")):
            images = images.to(device)
            masks_np = masks.cpu().numpy()
            preds = model(images)

            batch_size = images.shape[0]
            for i in range(batch_size):
                # ---------- 预处理 ----------
                file_idx   = batch_idx * batch_size + i
                file_name  = val_loader.dataset.img_files[file_idx]
                img_np     = images[i, 0].cpu().numpy()
                mask_np    = masks_np[i, 0]
                pred_np    = preds[i, 0].cpu().numpy()
                h          = pred_np.shape[0]
                half_h     = h // 2
                result_mask = np.zeros_like(pred_np)
                thresh_mask = np.zeros_like(pred_np)

                # ---------- 上下半区连通域 + 重心 ----------
                for zone, (slc, y_off) in enumerate(((slice(0, half_h), 0),
                                                     (slice(half_h, h), half_h))):
                    part = pred_np[slc, :]
                    cc   = part > prob_threshold
                    if np.any(cc):
                        lbl, n = ndimage.label(cc)
                        if n:
                            sizes         = ndimage.sum(cc, lbl, range(1, n+1))
                            max_lbl       = np.argmax(sizes) + 1
                            region        = (lbl == max_lbl)
                            thresh_mask[slc, :][region] = 1      # 阈值区域
                            ys, xs        = np.where(region)
                            cy, cx        = int(np.mean(ys)) + y_off, int(np.mean(xs))
                            result_mask[cy, cx] = 1              # 重心点

                # ---------- 取 GT 坐标 ----------
                gt_y, gt_x = np.where(mask_np == 1)
                upper_gt = (gt_y[gt_y < half_h], gt_x[gt_y < half_h])
                lower_gt = (gt_y[gt_y >= half_h], gt_x[gt_y >= half_h])

                # ---------- 绘图 ----------
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                # ① 原图 + 阈值区域 + GT
                axes[0].imshow(img_np, cmap='gray')
                overlay = np.zeros((*thresh_mask.shape, 4))
                overlay[thresh_mask > 0] = [0, 0, 1, 0.3]  # 蓝色半透明
                axes[0].imshow(overlay)
                if len(upper_gt[0]): axes[0].scatter(upper_gt[1], upper_gt[0], c='r', s=30, label='GT')
                if len(lower_gt[0]): axes[0].scatter(lower_gt[1], lower_gt[0], c='r', s=30)
                axes[0].set_title('Image + Threshold Regions + GT'); axes[0].axis('off')

                # ② 概率热图
                im = axes[1].imshow(pred_np, cmap='jet', vmin=0, vmax=1); plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                axes[1].set_title(f'Probability Map (>{prob_threshold})'); axes[1].axis('off')

                # ③ 原图 + 重心点
                axes[2].imshow(img_np, cmap='gray')
                axes[2].imshow(overlay)
                cy, cx = np.where(result_mask == 1)
                if len(cy): axes[2].scatter(cx, cy, c='y', marker='x', s=40, label='Centroid')
                if len(upper_gt[0]): axes[2].scatter(upper_gt[1], upper_gt[0], c='r', s=30, label='GT')
                if len(lower_gt[0]): axes[2].scatter(lower_gt[1], lower_gt[0], c='r', s=30)
                axes[2].set_title('Image + Centroid'); axes[2].axis('off')

                plt.suptitle(f'Sample {file_idx+1}/{len(val_loader.dataset)}  •  {file_name}')
                plt.tight_layout(); plt.show()

# 主函数
def main():
    # 超参数设置
    batch_size = 4
    distance_threshold = 3.0  # 距离阈值，用于筛选样本
    prob_threshold = 0.5  # 概率阈值，用于提取预测区域
    
    # 指定使用特定模型文件
    model_filename = "zygomatic_model_20250625_151231.pth"
    model_path = os.path.join(model_save_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"错误: 未找到指定的模型文件 {model_filename}。请确认模型文件存在。")
        return
    
    print(f"使用模型文件: {model_filename}")
    
    # 获取数据加载器
    val_loader = get_dataloader(batch_size)
    
    # 初始化模型
    model = UNet(n_channels=1, n_classes=1)
    
    # 加载训练好的模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型加载成功，使用设备: {device}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    model.to(device)
    
    # 计算平均距离
    print("\n计算验证集上的平均距离...")
    avg_upper_distance, avg_lower_distance, avg_total_distance = calculate_average_distances(model, val_loader)
    
    # 可视化距离大于设定阈值的样本
    print(f"\n可视化距离大于{distance_threshold}像素的样本，使用预测概率阈值{prob_threshold}...")
    visualize_large_distance_samples(model, val_loader, distance_threshold=distance_threshold, prob_threshold=prob_threshold)
    print("\n逐一可视化验证集所有样本 ...")
    visualize_all_samples(model, val_loader, prob_threshold=prob_threshold)
    # 可视化预测结果
    visualize_predictions(model, val_loader, num_samples=5)  # 增加样本数为5

if __name__ == "__main__":
    main() 