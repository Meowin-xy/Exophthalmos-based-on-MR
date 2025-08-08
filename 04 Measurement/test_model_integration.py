import os
import sys
import numpy as np
import nibabel as nib

# 添加当前目录到路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 使用importlib来导入以数字开头的模块
import importlib.util
spec = importlib.util.spec_from_file_location("validation_2d", "2dValidation.py")
validation_2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation_2d)

load_zygomatic_model = validation_2d.load_zygomatic_model
predict_quantu_positions = validation_2d.predict_quantu_positions

def test_model_loading():
    """
    测试模型加载功能
    """
    print("=== 测试模型加载 ===")
    
    # 模型路径 (你需要确保这个路径是正确的)
    model_path = r"../finalnewdata/saved_models/zygomatic_model_20250625_151231.pth"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保模型文件路径正确")
        return None, None
    
    model, device = load_zygomatic_model(model_path)
    
    if model is not None:
        print(f"模型加载成功！使用设备: {device}")
        return model, device
    else:
        print("模型加载失败")
        return None, None

def test_prediction():
    """
    测试预测功能
    """
    print("\n=== 测试预测功能 ===")
    
    model, device = test_model_loading()
    if model is None:
        return
    
    # 创建测试数据 (模拟320x112x1的图像)
    test_data = np.random.rand(320, 112, 1) * 255
    test_data = test_data.astype(np.float32)
    
    print(f"测试数据形状: {test_data.shape}")
    
    # 测试预测
    try:
        predictions = predict_quantu_positions(model, device, test_data, prob_threshold=0.5)
        print(f"预测成功！找到 {len(predictions)} 个颧突点:")
        for i, (x, y) in enumerate(predictions):
            print(f"  点 {i+1}: (x={x}, y={y})")
    except Exception as e:
        print(f"预测失败: {str(e)}")

def test_real_file():
    """
    测试真实文件处理
    """
    print("\n=== 测试真实文件处理 ===")
    
    model, device = test_model_loading()
    if model is None:
        return
    
    # 测试目录
    quantu_dir = r"C:\Users\xrVis001\Desktop\EyeData\Eyeball0714\Eyeball0714\Validation\2D\quantuSlice"
    
    if not os.path.exists(quantu_dir):
        print(f"测试目录不存在: {quantu_dir}")
        return
    
    # 找到第一个nii.gz文件进行测试
    nii_files = [f for f in os.listdir(quantu_dir) if f.endswith('.nii.gz')]
    
    if not nii_files:
        print("没有找到nii.gz文件")
        return
    
    test_file = os.path.join(quantu_dir, nii_files[0])
    print(f"测试文件: {test_file}")
    
    try:
        # 加载文件
        img = nib.load(test_file)
        data = img.get_fdata()
        print(f"文件数据形状: {data.shape}")
        
        # 进行预测
        predictions = predict_quantu_positions(model, device, data, prob_threshold=0.5)
        print(f"预测成功！找到 {len(predictions)} 个颧突点:")
        for i, (x, y) in enumerate(predictions):
            print(f"  点 {i+1}: (x={x}, y={y})")
            
    except Exception as e:
        print(f"处理真实文件时出错: {str(e)}")

if __name__ == "__main__":
    print("颧突检测模型集成测试")
    print("=" * 50)
    
    # 运行测试
    test_model_loading()
    test_prediction()
    test_real_file()
    
    print("\n测试完成！") 