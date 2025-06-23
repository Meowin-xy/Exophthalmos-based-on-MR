# predict.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib  
import os
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet34_Weights
import pandas as pd
from scipy.spatial.distance import euclidean
import re
import torch.nn.functional as F

class NMRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
        self.labels = {self.extract_id(f): f for f in os.listdir(label_dir) if f.endswith('.nii.gz')}
        self.transform = transform

    def extract_id(self, filename):
        # 使用正则表达式提取ID（例如从 "83899659-2024071-57.nii.gz" 中提取 "83899659-2024071-57"）
        match = re.search(r'(.+)\.nii\.gz', filename)
        return match.group(1) if match else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_id = self.extract_id(img_filename)
        label_filename = self.labels.get(img_id)

        if label_filename is None:
            raise FileNotFoundError(f"No label found for image ID {img_id}")

        img_path = os.path.join(self.image_dir, img_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Handle 3D data (assuming taking a 2D slice)
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2] // 2]
            label = label[:, :, label.shape[2] // 2]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        image = image.to(dtype=torch.float32)
        label = label.to(dtype=torch.float32)

        return image, label, img_id  # 返回三个值

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 120), antialias=True)
])

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels + skip_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, concat_with):
        x = self.up(x)
        # Adjust size before concatenation
        x = F.interpolate(x, size=concat_with.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, concat_with], dim=1)
        x = self.conv_relu(x)
        return x


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Load a pre-trained ResNet model
        self.base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Extract layers from the pre-trained ResNet model
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool
        )
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        # ... [rest of the encoder layers as before] ...

        # Decoder layers
        # Adjust the channels according to your network's architecture
        self.decoder4 = DecoderBlock(512, 256, 256, 256)  # The last number is for skip connection channels
        self.decoder3 = DecoderBlock(256, 128, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 32, 64)  # The last number should match the output channels of layer0

        self.final_conv = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        # Encoder path
        x0 = self.layer0(x)  # Initial convolutional layer
        x1 = self.layer1(x0)  # First ResNet layer
        x2 = self.layer2(x1)  # Second ResNet layer
        x3 = self.layer3(x2)  # Third ResNet layer
        x4 = self.layer4(x3)  # Fourth ResNet layer

        # Decoder path
        # Each step in the decoder path upsamples the feature map and concatenates 
        # it with the corresponding feature map from the encoder path (skip connection)
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        # Final upsampling to match input size
        d1 = F.interpolate(d1, size=(320, 120), mode='bilinear', align_corners=False)

        out = self.final_conv(d1)
        return out


# 配置路径
image_dir = r'F:\gxy2025\tao-cal\split_data\validation\images'
label_dir = r'F:\gxy2025\tao-cal\split_data\validation\labels'
output_dir = r'F:\gxy2025\tao-cal\predict'
weight_path = r'F:\gxy2025\tao-cal\best_unet_model_fold_3.pth'

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(n_class=1).to(device)
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
model.eval()

# 创建数据集
dataset = NMRDataset(image_dir, label_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 预测流程
distances = []
with torch.no_grad():
    for images, labels, img_ids in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 模型预测
        outputs = torch.sigmoid(model(images))
        
        # 调试输出
        print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
        
        # 调整阈值（示例使用动态阈值）
        threshold = 0.95


        binary_outputs = (outputs > threshold).float()

        # 处理每个样本
        for i in range(binary_outputs.shape[0]):
            img_id = img_ids[i]
            pred = binary_outputs[i].squeeze().cpu().numpy()
            label = labels[i].squeeze().cpu().numpy()

            # 保存预测结果
            pred_nii = nib.Nifti1Image(pred.astype(np.float32), np.eye(4))
            pred_path = os.path.join(output_dir, f'{img_id}_pred.nii.gz')
            nib.save(pred_nii, pred_path)

            # 计算距离指标
            if pred.sum() > 0 and label.sum() > 0:
                pred_coords = np.argwhere(pred > 0)
                true_coords = np.argwhere(label > 0)
                min_dist = min(euclidean(p, t) for p in pred_coords for t in true_coords)
            else:
                min_dist = float('inf')
            
            distances.append((img_id, min_dist))

# 保存结果
pd.DataFrame(distances, columns=['Image ID', 'Min Distance']).to_excel(
    os.path.join(output_dir, 'distances.xlsx'), index=False
)

print("Prediction completed.")