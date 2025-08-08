import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

# 添加U-Net模型定义 (来自Big3DEvaluating.py)
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

def load_zygomatic_model(model_path):
    """
    加载颧突检测模型
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"颧突检测模型加载成功，使用设备: {device}")
        return model, device
    except Exception as e:
        print(f"加载颧突检测模型时出错: {e}")
        return None, None

def predict_zygomatic_positions(model, device, image_slice, prob_threshold=0.5):
    """
    使用模型预测颧突位置
    """
    if model is None:
        return []
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(image_slice).float().unsqueeze(0).unsqueeze(0)  # 添加batch和通道维度
    img_tensor = transform(img_tensor)
    
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        prediction = model(img_tensor)
        pred_np = prediction[0, 0].cpu().numpy()  # 转回numpy
        
        height = pred_np.shape[0]
        half_height = height // 2
        
        # 将图像分为上下两半区
        upper_half = pred_np[:half_height, :]
        lower_half = pred_np[half_height:, :]
        
        predicted_points = []
        
        # 处理上半区域
        upper_threshold = upper_half > prob_threshold
        if np.any(upper_threshold):
            labeled_upper, num_features_upper = ndimage.label(upper_threshold)
            
            if num_features_upper > 0:
                sizes_upper = ndimage.sum(upper_threshold, labeled_upper, range(1, num_features_upper + 1))
                max_label_upper = np.argmax(sizes_upper) + 1
                max_region_upper = (labeled_upper == max_label_upper)
                
                upper_y, upper_x = np.where(max_region_upper)
                if len(upper_y) > 0:
                    upper_center_y = int(np.mean(upper_y))
                    upper_center_x = int(np.mean(upper_x))
                    predicted_points.append((upper_center_x, upper_center_y))
        
        # 处理下半区域
        lower_threshold = lower_half > prob_threshold
        if np.any(lower_threshold):
            labeled_lower, num_features_lower = ndimage.label(lower_threshold)
            
            if num_features_lower > 0:
                sizes_lower = ndimage.sum(lower_threshold, labeled_lower, range(1, num_features_lower + 1))
                max_label_lower = np.argmax(sizes_lower) + 1
                max_region_lower = (labeled_lower == max_label_lower)
                
                lower_y, lower_x = np.where(max_region_lower)
                if len(lower_y) > 0:
                    lower_center_y = int(np.mean(lower_y)) + half_height
                    lower_center_x = int(np.mean(lower_x))
                    predicted_points.append((lower_center_x, lower_center_y))
        
        return predicted_points

def get_quantu_positions_with_model(quantu_slice_path, model, device, best_slice_idx):
    """
    使用模型预测颧突位置，替代从mask文件读取
    """
    try:
        # Load nii.gz file
        img = nib.load(quantu_slice_path)
        data = img.get_fdata()
        
        # Get voxel size information from header
        voxel_sizes = img.header.get_zooms()
        pixel_spacing_x = voxel_sizes[0]  # mm per pixel in x direction
        pixel_spacing_y = voxel_sizes[1]  # mm per pixel in y direction
        if len(voxel_sizes) > 2:
            pixel_spacing_z = voxel_sizes[2]  # mm per pixel in z direction
        else:
            pixel_spacing_z = 1.0
        
        print(f"  Quantu slice shape: {data.shape}")
        print(f"  Using slice index: {best_slice_idx}")
        print(f"  Voxel sizes: x={pixel_spacing_x:.3f}mm, y={pixel_spacing_y:.3f}mm, z={pixel_spacing_z:.3f}mm")
        
        # 获取指定层的图像切片
        if len(data.shape) == 3:
            if best_slice_idx < data.shape[2]:
                image_slice = data[:, :, best_slice_idx]
            else:
                print(f"  Warning: best_slice_idx {best_slice_idx} >= data.shape[2] {data.shape[2]}, using last slice")
                image_slice = data[:, :, -1]
        else:
            # 如果是2D数据，直接使用
            image_slice = data
        
        print(f"  Image slice shape for prediction: {image_slice.shape}")
        
        # 使用模型预测颧突位置
        predicted_points = predict_zygomatic_positions(model, device, image_slice)
        
        print(f"  Model predicted {len(predicted_points)} zygomatic points:")
        for i, point in enumerate(predicted_points):
            print(f"    Point {i+1}: (x={point[0]}, y={point[1]}, z={best_slice_idx})")
        
        # 将2D坐标转换为3D坐标 (x, y, z)
        point_coords_3d = []
        for point in predicted_points:
            coord_3d = (point[0], point[1], best_slice_idx)
            point_coords_3d.append(coord_3d)
        
        return point_coords_3d, (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
    except Exception as e:
        print(f"Error processing quantu slice file {quantu_slice_path}: {str(e)}")
        return [], (1.0, 1.0, 1.0)

def get_quantu_positions(file_path):
    """
    从mask文件中获取真实颧突位置
    """
    try:
        # Load nii.gz file
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Get voxel size information from header
        voxel_sizes = img.header.get_zooms()
        pixel_spacing_x = voxel_sizes[0]  # mm per pixel in x direction
        pixel_spacing_y = voxel_sizes[1]  # mm per pixel in y direction
        if len(voxel_sizes) > 2:
            pixel_spacing_z = voxel_sizes[2]  # mm per pixel in z direction
        else:
            pixel_spacing_z = 1.0
        
        # Find coordinates where value equals 1
        coords = np.where(data == 1)
        
        if len(coords[0]) == 0:
            return [], (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
        # Store coordinates for this file
        point_coords = []
        
        for i in range(len(coords[0])):
            if len(coords) == 3:  # 3D data
                coord = (coords[1][i], coords[0][i], coords[2][i])  # (x, y, z)
            else:  # 2D data
                coord = (coords[1][i], coords[0][i])  # (x, y)
            
            point_coords.append(coord)
        
        return point_coords, (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
    except Exception as e:
        print(f"Error processing quantu file {file_path}: {str(e)}")
        return [], (1.0, 1.0, 1.0)

def extract_numbers_from_filename(filename):
    """
    Extract all numbers from filename and return as a tuple for matching
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.endswith('.nii'):
        name_without_ext = os.path.splitext(name_without_ext)[0]
    
    # Extract all numbers from filename
    numbers = re.findall(r'\d+', name_without_ext)
    return tuple(numbers)

def find_matching_files(quantu_slice_files, quantu_mask_files):
    """
    Find matching files based on numeric parts in filenames
    """
    quantu_slice_dict = {}
    quantu_mask_dict = {}
    
    # Build dictionaries with number patterns as keys
    for file in quantu_slice_files:
        numbers = extract_numbers_from_filename(file)
        if numbers:
            quantu_slice_dict[numbers] = file
    
    for file in quantu_mask_files:
        numbers = extract_numbers_from_filename(file)
        if numbers:
            quantu_mask_dict[numbers] = file
    
    # Find common number patterns
    common_numbers = set(quantu_slice_dict.keys()).intersection(set(quantu_mask_dict.keys()))
    
    # Return pairs of matching files
    matching_pairs = []
    for numbers in common_numbers:
        matching_pairs.append((quantu_slice_dict[numbers], quantu_mask_dict[numbers]))
    
    return matching_pairs



def process_corresponding_files(quantu_slice_directory, quantu_mask_directory, output_directory, model, device):
    """
    处理对应文件，计算模型预测位置和真实位置之间的距离
    """
    # Check if directories exist
    if not os.path.exists(quantu_slice_directory):
        print(f"Quantu slice directory does not exist: {quantu_slice_directory}")
        return
    
    if not os.path.exists(quantu_mask_directory):
        print(f"Quantu mask directory does not exist: {quantu_mask_directory}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    # Get all nii.gz files from both directories
    quantu_slice_files = [f for f in os.listdir(quantu_slice_directory) if f.endswith('.nii.gz')]
    quantu_mask_files = [f for f in os.listdir(quantu_mask_directory) if f.endswith('.nii.gz')]
    
    # Find matching files based on numeric parts
    matching_pairs = find_matching_files(quantu_slice_files, quantu_mask_files)
    
    if not matching_pairs:
        print("No matching files found between the two directories based on numeric parts")
        return
    
    print(f"Found {len(matching_pairs)} matching file pairs to process")
    print("="*60)
    
    # Prepare data for CSV output
    csv_data = []
    
    for quantu_slice_file, quantu_mask_file in sorted(matching_pairs):
        print(f"Processing: {quantu_slice_file} <-> {quantu_mask_file}")
        
        quantu_slice_path = os.path.join(quantu_slice_directory, quantu_slice_file)
        quantu_mask_path = os.path.join(quantu_mask_directory, quantu_mask_file)
        
        # 获取真实颧突位置
        true_positions, true_voxel_sizes = get_quantu_positions(quantu_mask_path)
        
        # 使用模型预测颧突位置（使用切片中心作为best_slice_idx）
        # 先获取切片文件的形状来确定best_slice_idx
        try:
            img = nib.load(quantu_slice_path)
            data = img.get_fdata()
            if len(data.shape) == 3:
                best_slice_idx = data.shape[2] // 2  # 使用中间层
            else:
                best_slice_idx = 0
        except:
            best_slice_idx = 0
        
        predicted_positions, pred_voxel_sizes = get_quantu_positions_with_model(quantu_slice_path, model, device, best_slice_idx)
        
        print(f"  Found {len(true_positions)} true zygomatic points")
        print(f"  Found {len(predicted_positions)} predicted zygomatic points")
        
        # 计算左右眼的颧突距离
        left_eye_distance = np.nan
        right_eye_distance = np.nan
        
        if len(predicted_positions) >= 2 and len(true_positions) >= 2:
            # 根据x坐标排序预测点和真实点（左到右）
            sorted_pred = sorted(predicted_positions, key=lambda p: p[0])
            sorted_true = sorted(true_positions, key=lambda p: p[0])
            
            # 左眼（x坐标较小的点）
            left_pred = sorted_pred[0]
            left_true = sorted_true[0]
            left_eye_distance = np.sqrt(
                ((left_pred[0] - left_true[0]) * true_voxel_sizes[0])**2 + 
                ((left_pred[1] - left_true[1]) * true_voxel_sizes[1])**2 + 
                ((left_pred[2] - left_true[2]) * true_voxel_sizes[2])**2 if len(left_pred) > 2 and len(left_true) > 2 else 0
            )
            
            # 右眼（x坐标较大的点）
            right_pred = sorted_pred[-1]
            right_true = sorted_true[-1]
            right_eye_distance = np.sqrt(
                ((right_pred[0] - right_true[0]) * true_voxel_sizes[0])**2 + 
                ((right_pred[1] - right_true[1]) * true_voxel_sizes[1])**2 + 
                ((right_pred[2] - right_true[2]) * true_voxel_sizes[2])**2 if len(right_pred) > 2 and len(right_true) > 2 else 0
            )
            
            print(f"  Left eye zygomatic distance: {left_eye_distance:.2f} mm")
            print(f"  Right eye zygomatic distance: {right_eye_distance:.2f} mm")
        elif len(predicted_positions) == 1 and len(true_positions) >= 1:
            # 只有一个预测点，找最近的真实点
            pred_point = predicted_positions[0]
            min_distance = float('inf')
            for true_point in true_positions:
                distance = np.sqrt(
                    ((pred_point[0] - true_point[0]) * true_voxel_sizes[0])**2 + 
                    ((pred_point[1] - true_point[1]) * true_voxel_sizes[1])**2 + 
                    ((pred_point[2] - true_point[2]) * true_voxel_sizes[2])**2 if len(pred_point) > 2 and len(true_point) > 2 else 0
                )
                if distance < min_distance:
                    min_distance = distance
            
            # 根据预测点的x坐标判断是左眼还是右眼
            # 这里需要一个阈值，假设图像中心线
            if len(true_positions) >= 2:
                sorted_true = sorted(true_positions, key=lambda p: p[0])
                center_x = (sorted_true[0][0] + sorted_true[-1][0]) / 2
                if pred_point[0] < center_x:
                    left_eye_distance = min_distance
                    print(f"  Left eye zygomatic distance: {left_eye_distance:.2f} mm")
                else:
                    right_eye_distance = min_distance
                    print(f"  Right eye zygomatic distance: {right_eye_distance:.2f} mm")
            else:
                left_eye_distance = min_distance
                print(f"  Single point distance (assigned to left): {left_eye_distance:.2f} mm")
        else:
            print(f"  Warning: Insufficient points for distance calculation")
            print(f"  Predicted: {len(predicted_positions)}, True: {len(true_positions)}")
        
        # 准备简化的CSV数据
        row_data = {
            'filename': quantu_slice_file,
            'left_eye_distance_mm': left_eye_distance,
            'right_eye_distance_mm': right_eye_distance
        }
        
        csv_data.append(row_data)
        print("-" * 40)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    output_file = os.path.join(output_directory, 'zygomatic_left_right_eye_distances.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total files processed: {len(csv_data)}")
    print("\nLeft and Right Eye Distance Summary:")
    print(df[['left_eye_distance_mm', 'right_eye_distance_mm']].describe())

# Main execution
if __name__ == "__main__":
    # Define directories
    quantu_slice_directory = r"C:\Users\xrVis001\Desktop\EyeData\Eyeball0714\Eyeball0714\Validation\3D\quantuSlice"
    quantu_mask_directory = r"C:\Users\xrVis001\Desktop\EyeData\Eyeball0714\Eyeball0714\Validation\3D\quantuMask"
    output_directory = r"C:\Users\xrVis001\Desktop\EyeData\Eyeball0714\Eyeball0714\Validation\3D"
    
    # Load the zygomatic detection model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "final3Ddata", "saved_models", "zygomatic_model_20250716_101204.pth")
    
    print(f"尝试加载模型: {model_path}")
    model, device = load_zygomatic_model(model_path)
    
    if model is None:
        print("颧突检测模型加载失败，请检查模型文件路径。")
        print("请确保模型文件存在于 final3Ddata/saved_models/ 目录中")
        exit()
    
    # Process corresponding files and generate validation results
    print("=== 处理文件并生成验证结果 ===")
    print("计算模型预测的颧突位置和真实颧突位置之间的距离...")
    print(f"Quantu slice directory (input images): {quantu_slice_directory}")
    print(f"Quantu mask directory (ground truth): {quantu_mask_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Model path: {model_path}")
    print("="*60)
    
    process_corresponding_files(quantu_slice_directory, quantu_mask_directory, output_directory, model, device)
