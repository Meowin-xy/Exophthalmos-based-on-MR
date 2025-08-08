import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from tqdm import tqdm

# 数据路径设置
data_dir = r"C:\Users\xrVis001\Desktop\tao-cal\split_data"
train_img_dir = os.path.join(data_dir, "train", "images")
train_label_dir = os.path.join(data_dir, "train", "labels")
val_img_dir = os.path.join(data_dir, "validation", "images")
val_label_dir = os.path.join(data_dir, "validation", "labels")

# 创建模型保存目录
model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(model_save_dir, exist_ok=True)

# 自定义数据集类
class ZygomaticDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))
        
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
        
        # 只取第一个切片(维度是320x120x1)
        img_slice = img_data[:, :, 0]
        label_slice = label_data[:, :, 0]
        
        # 将标签二值化 - 颧突位置为1，其余为0
        binary_mask = np.zeros_like(label_slice)
        if np.max(label_slice) > 0:  # 确保有标签
            # 假设颧突标记为特定值
            # 这里需要根据实际标签调整
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
def get_dataloaders(batch_size=4):
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
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model.to(device)
    
    # 跟踪训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 模型保存路径
    model_save_path = os.path.join(model_save_dir, "best_zygomatic_model.pth")
    
    for epoch in range(num_epochs):
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
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型到: {model_save_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, train_losses, val_losses

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
        
        # 修改预测结果处理方式 - 只保留上下半区最高值点
        modified_predictions = []
        for pred in predictions:
            pred_np = pred[0].cpu().numpy()  # 转为numpy数组处理
            height = pred_np.shape[0]
            half_height = height // 2
            
            # 将图像分为上下两半区
            upper_half = pred_np[:half_height, :]
            lower_half = pred_np[half_height:, :]
            
            # 创建空白掩码
            result_mask = np.zeros_like(pred_np)
            
            # 找到上半区最大值位置
            if np.max(upper_half) > 0:  # 确保有预测值
                upper_max_idx = np.unravel_index(upper_half.argmax(), upper_half.shape)
                # 在原图对应位置标记为1
                result_mask[upper_max_idx[0], upper_max_idx[1]] = 1
            
            # 找到下半区最大值位置
            if np.max(lower_half) > 0:  # 确保有预测值
                lower_max_idx = np.unravel_index(lower_half.argmax(), lower_half.shape)
                # 注意调整索引到原图对应位置
                result_mask[lower_max_idx[0] + half_height, lower_max_idx[1]] = 1
            
            modified_predictions.append(result_mask)
        
        # 转换为正确的形状以便显示
        predictions = np.array(modified_predictions)[:, np.newaxis, :, :]
    
    # 转回CPU进行可视化
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    # 显示结果
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # 原始图像与真实标签叠加
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].imshow(masks[i, 0], cmap='jet', alpha=0.5)  # 使用alpha控制透明度
        axes[i, 0].set_title('Image + Ground Truth')
        axes[i, 0].axis('off')
        
        # 原始图像与预测结果叠加
        axes[i, 1].imshow(images[i, 0], cmap='gray')
        axes[i, 1].imshow(predictions[i, 0], cmap='jet', alpha=0.5)  # 使用alpha控制透明度
        axes[i, 1].set_title('Image + Prediction')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 超参数设置
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(batch_size)
    
    # 初始化模型
    model = UNet(n_channels=1, n_classes=1)
    
    # 定义损失函数和优化器
    # 使用二元交叉熵损失，适合分割任务
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    
    # 加载最佳模型
    model_load_path = os.path.join(model_save_dir, "best_zygomatic_model.pth")
    model.load_state_dict(torch.load(model_load_path))
    
    # 评估模型
    precision, recall, f1 = evaluate_model(model, val_loader)
    
    # 可视化预测结果
    visualize_predictions(model, val_loader)

if __name__ == "__main__":
    main()
