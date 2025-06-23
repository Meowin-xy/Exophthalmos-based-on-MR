import torch

def compare_pth_files_simple(file1, file2):
    # 加载两个.pth文件
    state_dict1 = torch.load(file1)
    state_dict2 = torch.load(file2)

    # 检查两个状态字典的键是否一致
    if state_dict1.keys() != state_dict2.keys():
        print("两个.pth文件的键不一致！")
        missing_in_file1 = set(state_dict2.keys()) - set(state_dict1.keys())
        missing_in_file2 = set(state_dict1.keys()) - set(state_dict2.keys())
        if missing_in_file1:
            print(f"仅在 {file2} 中存在的键: {missing_in_file1}")
        if missing_in_file2:
            print(f"仅在 {file1} 中存在的键: {missing_in_file2}")
        return

    # 逐层检查形状是否一致
    shape_differences_found = False
    for key in state_dict1.keys():
        if state_dict1[key].shape != state_dict2[key].shape:
            shape_differences_found = True
            print(f"权重 '{key}' 的形状不一致:")
            print(f"  {file1} 中的形状: {state_dict1[key].shape}")
            print(f"  {file2} 中的形状: {state_dict2[key].shape}")

    if not shape_differences_found:
        print("两个.pth文件的权重形状完全一致！")


# 示例调用
file1 = r"G:\finaldata0528\Code\Segment\2DEyeball\save_models\model_epoch_best.pth"
file2 = r"G:\finaldata0528\Code\Segment\2DEyeball\models\model_epoch_best.pth"
compare_pth_files_simple(file1, file2)