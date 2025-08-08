import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from tqdm import tqdm
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.ndimage
import shutil

# 数据路径设置
data_img_dir = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\2DMRI\Quantu\quantuSlicemea"
data_label_dir = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\2DMRI\Quantu\quantuMaskmea"

# 创建分割后的数据目录
split_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2Dsplit_data")
train_img_dir = os.path.join(split_data_dir, "train", "images")
train_label_dir = os.path.join(split_data_dir, "train", "labels")
val_img_dir = os.path.join(split_data_dir, "validation", "images")
val_label_dir = os.path.join(split_data_dir, "validation", "labels")

# 创建模型保存目录
model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(model_save_dir, exist_ok=True)

# 划分数据集函数
def split_dataset(img_dir, label_dir, train_ratio=0.8, random_seed=42):
    # 确保目录存在
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 获取所有图像文件，过滤掉以"._"开头的隐藏文件
    img_files = sorted([f for f in os.listdir(img_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and not f.startswith('._')])
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 随机打乱文件顺序
    np.random.shuffle(img_files)
    
    # 计算训练集大小
    train_size = int(len(img_files) * train_ratio)
    
    # 划分训练集和验证集
    train_files = img_files[:train_size]
    val_files = img_files[train_size:]
    
    print(f"划分数据集: {len(train_files)}个训练样本, {len(val_files)}个验证样本")
    
    # 复制文件到对应目录
    for file in train_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(train_img_dir, file))
        if os.path.exists(os.path.join(label_dir, file)):
            shutil.copy(os.path.join(label_dir, file), os.path.join(train_label_dir, file))
        else:
            print(f"警告: 标签文件 {file} 不存在于 {label_dir}")
        
    for file in val_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(val_img_dir, file))
        if os.path.exists(os.path.join(label_dir, file)):
            shutil.copy(os.path.join(label_dir, file), os.path.join(val_label_dir, file))
        else:
            print(f"警告: 标签文件 {file} 不存在于 {label_dir}")
    
    return train_files, val_files

# 自定义数据集类
class ZygomaticDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and not f.startswith('._')])
        
    def __len__(self):
        return len(self.img_files)
    
    def create_circle_mask(self, shape, center, radius):
        """在给定形状的掩码上创建一个半径为radius的圆"""
        y, x = np.ogrid[:shape[0], :shape[1]]
        # 计算每个点到圆心的欧氏距离
        dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        # 创建圆形掩码，距离≤radius的点设为1，其余为0
        mask = (dist_from_center <= radius).astype(np.float32)
        return mask
    
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
            # 找到所有值大于0的点（颧突标记点）
            points = np.argwhere(label_slice > 0)
            
            # 为每个点创建半径为5的圆
            for point in points:
                circle_mask = self.create_circle_mask(label_slice.shape, point, radius=5)
                # 将圆添加到二进制掩码
                binary_mask = np.logical_or(binary_mask, circle_mask).astype(np.float32)
        
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
def get_dataloaders(batch_size=4):
    # 首先检查是否已经分割了数据集
    if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
        print("正在划分数据集...")
        split_dataset(data_img_dir, data_label_dir)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集
    train_dataset = ZygomaticDataset(train_img_dir, train_label_dir, transform=transform)
    val_dataset = ZygomaticDataset(val_img_dir, val_label_dir, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100, patience=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model.to(device)
    
    # 跟踪训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 早停机制变量
    counter = 0
    early_stop = False
    
    # 获取当前时间作为模型文件名的一部分
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 模型保存路径
    model_save_path = os.path.join(model_save_dir, f"zygomatic_model_{timestamp}.pth")
    
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        # 训练模式
        model.train()
        running_loss = 0.0
        
        for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证模式
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, masks)
                
                running_val_loss += val_loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # 学习率调整
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型到: {model_save_path}")
            counter = 0
        else:
            counter += 1
            print(f"验证损失未改善。 计数器: {counter}/{patience}")
            if counter >= patience:
                early_stop = True
                print(f"早停触发于第{epoch+1}轮")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, train_losses, val_losses, model_save_path

# 评估函数
def evaluate_model(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 二值化预测 (阈值0.5)
            preds = (outputs > 0.5).float().cpu().numpy()
            masks = masks.cpu().numpy()
            
            all_preds.extend(preds.reshape(-1))
            all_masks.extend(masks.reshape(-1))
    
    # 计算指标
    precision = precision_score(all_masks, all_preds, zero_division=0)
    recall = recall_score(all_masks, all_preds, zero_division=0)
    f1 = f1_score(all_masks, all_preds, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1

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
        
        # 二值化预测结果 (阈值0.5)
        binary_predictions = (predictions > 0.5).float()
        
        # 转回CPU进行可视化
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        predictions = predictions.cpu().numpy()
        binary_predictions = binary_predictions.cpu().numpy()
    
    # 显示结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # 原始图像
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # 原始图像与真实标签叠加
        axes[i, 1].imshow(images[i, 0], cmap='gray')
        axes[i, 1].imshow(masks[i, 0], cmap='jet', alpha=0.5)  # 使用alpha控制透明度
        axes[i, 1].set_title('Ground Truth Circular Mask')
        axes[i, 1].axis('off')
        
        # 原始图像与预测结果叠加
        axes[i, 2].imshow(images[i, 0], cmap='gray')
        axes[i, 2].imshow(binary_predictions[i, 0], cmap='jet', alpha=0.5)
        axes[i, 2].set_title('Model Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 超参数设置
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 150  # 增加训练轮数
    patience = 10     # 早停耐心值
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(batch_size)
    
    # 初始化模型
    model = UNet(n_channels=1, n_classes=1)
    
    # 定义损失函数和优化器
    # 使用二元交叉熵损失，适合分割任务
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    model, train_losses, val_losses, best_model_path = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 评估模型
    precision, recall, f1 = evaluate_model(model, val_loader)
    
    # 可视化预测结果
    visualize_predictions(model, val_loader)

if __name__ == "__main__":
    main()
