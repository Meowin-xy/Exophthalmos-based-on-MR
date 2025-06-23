import os
import numpy as np
import torch
import SimpleITK as sitk
from models import UNet,transforms
import matplotlib.pyplot as plt

def load_nii(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    return data, img.GetOrigin(), img.GetSpacing(), img.GetDirection()


def save_nii(data, output, origin, spacing, direction, save_path):
    save_nii_path = os.path.join(save_path)
    if not os.path.exists(save_nii_path):
        os.makedirs(save_nii_path)

    output = (output > 0.5).astype(np.int16)
    
    data_nii = sitk.GetImageFromArray(data.astype(np.float32))
    output_nii = sitk.GetImageFromArray(output.astype(np.float32))

    data_nii.SetOrigin(origin)
    data_nii.SetSpacing(spacing)
    data_nii.SetDirection(direction)
    sitk.WriteImage(data_nii, os.path.join(save_nii_path, 'image.nii.gz'))
    
    output_nii.SetOrigin(origin)
    output_nii.SetSpacing(spacing)
    output_nii.SetDirection(direction)
    sitk.WriteImage(output_nii, os.path.join(save_nii_path, 'pred.nii.gz'))

def visualize_slices(data, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [data.shape[0] // 4, data.shape[0] // 2, 3 * data.shape[0] // 4]
    for ax, slice_idx in zip(axes, slices):
        ax.imshow(data[slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def main():
    # 设置参数
    model_path = r"G:\finaldata0528\Code\Segment\3DEyeball\model_epoch_best.pth"
    input_folder = r"G:\finaldata0528\2506_3Dcrop"  # 修改为文件夹路径
    output_folder = r"G:\finaldata0528"

    # 加载模型（只需加载一次）
    model = UNet.BaselineUNet(1, 2, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 数据预处理管道
    test_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    # 获取所有NIfTI文件
    nii_files = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]
    print(f"Found {len(nii_files)} files to process")

    # 批量处理文件
    for filename in nii_files:
        file_path = os.path.join(input_folder, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            # 加载数据
            data, origin, spacing, direction = load_nii(file_path)
            print(f"Data shape: {data.shape}, Type: {data.dtype}")
            
            # 数据标准化
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = np.expand_dims(data, axis=-1)  # 添加通道维度

            # 预处理
            transformed_data = test_transforms({'input': data})['input']
            data_tensor = torch.unsqueeze(transformed_data, 0).to(device)

            # 推理预测
            with torch.no_grad():
                output = model(data_tensor)
                if output.shape[1] == 2:
                    output_prob = torch.softmax(output, dim=1)[:, 1].squeeze().cpu().numpy()
                else:
                    output_prob = output.squeeze().cpu().numpy()

            # 阈值处理与保存
            thresholds = [0.3, 0.4, 0.5]
            for threshold in thresholds:
                output_bin = (output_prob > threshold).astype(np.int16)
                file_output_dir = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_thresh{threshold}")
                save_nii(data[..., 0], output_bin, origin, spacing, direction, file_output_dir)
                print(f"Saved threshold {threshold} results to {file_output_dir}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    main()


