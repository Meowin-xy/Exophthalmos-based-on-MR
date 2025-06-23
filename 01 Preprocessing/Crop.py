# -*- coding: utf-8 -*-
# @Time    : 2025/06/23 
# @Author  : Xinyi Gou
# Note: This code is used for cropping both 2D and 3D MRI eyeballs and zygomatic processes

import os
import nibabel as nib
import numpy as np

# Configuration
image_input_dir = r'G:\2506_3Dtraniftistandardize'
mask_input_dir = '/Volumes/Pandas/finaldata/'  # When set to None, only crop images (skip mask cropping)
image_output_dir = r'G:\2506_3Dtraniftistandardizecrop'
mask_output_dir = '/Volumes/Pandas/finaldata/' # None

os.makedirs(image_output_dir, exist_ok=True)
#os.makedirs(mask_output_dir, exist_ok=True)

TARGET_SIZE = (320, 112, 80)  # 2D: (320, 112, 20); 3D: (320, 112, 80). For single-layer zygomatic process, change 20/80 to 1
THRESHOLD = 0  

def find_first_value_above_threshold(data, threshold=0):
    """Find first coordinates exceeding threshold in 3D slice"""
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[x, y] > threshold:
                return x, y
    return None, None

def crop_volume(data, target_size):
    """Center-crop 3D volume to specified dimensions"""
    if any(o < t for o, t in zip(data.shape, target_size)):
        raise ValueError(f"Dimensions {data.shape} smaller than target {target_size}")

    slices = [slice((d - t) // 2, (d - t) // 2 + t)
              for d, t in zip(data.shape, target_size)]
    return data[tuple(slices)]

def pad_volume(data, target_size):
    """Pad 3D volume to specified dimensions along Y-axis"""
    pad_y = max(0, target_size[1] - data.shape[1])
    if pad_y > 0:
        # Add all padding to the end (y-axis direction)
        pad_before = 0
        pad_after = pad_y  # All padding goes to posterior side
        
        data = np.pad(data, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
    return data

def process_image_pair(image_path, mask_path=None):
    """Process paired image and mask with synchronized cropping and padding"""
    try:
        # Load image
        img = nib.load(image_path)
        image_data = img.get_fdata()
        affine = img.affine.copy()
        header = img.header

        # Load mask if provided
        mask_data = None
        if mask_path:
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata()

        # Stage 1: Y-axis cropping (determined from image)
        z_mid = image_data.shape[2] // 2
        slice_data = image_data[:, :, z_mid]
        x, y = find_first_value_above_threshold(slice_data, THRESHOLD)
        if y is None:
            raise ValueError("No significant signal found in image")

        if y - 20 > 0:
            y_start = y - 20
            y_end = y + 92
        else:
            y_start = max(0, y)
            y_end = min(image_data.shape[1], y + 112)

        # Apply cropping to image
        image_stage1 = image_data[:, y_start:y_end, :]
        if mask_data is not None:
            mask_stage1 = mask_data[:, y_start:y_end, :]

        # Pad the image if Y-axis is less than 112
        image_stage1 = pad_volume(image_stage1, TARGET_SIZE)
        if mask_data is not None:
            mask_stage1 = pad_volume(mask_stage1, TARGET_SIZE)

        # Update affine for Y-crop
        offset_voxels = np.array([0, y_start, 0])
        affine[:3, 3] += affine[:3, :3] @ offset_voxels

        # Stage 2: Center cropping
        image_stage2 = crop_volume(image_stage1, TARGET_SIZE)
        if mask_data is not None:
            mask_stage2 = crop_volume(mask_stage1, TARGET_SIZE)

        # Update affine for center crop
        offset_voxels = np.array([(d - t) // 2 for d, t in zip(image_stage1.shape, TARGET_SIZE)])
        affine[:3, 3] += affine[:3, :3] @ offset_voxels

        # Save image
        base_name = os.path.basename(image_path)
        nib.save(nib.Nifti1Image(image_stage2, affine, header),
                 os.path.join(image_output_dir, base_name))

        # Save mask if available
        if mask_data is not None:
            nib.save(nib.Nifti1Image(mask_stage2, affine, header),
                     os.path.join(mask_output_dir, base_name))

        return True

    except Exception as e:
        print(f"Error processing pair {os.path.basename(image_path)}: {str(e)}")
        return False

# Process all pairs
for filename in os.listdir(image_input_dir):
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(image_input_dir, filename)
        mask_path = os.path.join(mask_input_dir, filename) if os.path.exists(os.path.join(mask_input_dir, filename)) else None

        success = process_image_pair(image_path, mask_path)
        status = "Success" if success else "Failed"
        print(f"{status}: {filename}")
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(image_input_dir, filename)
        mask_path = os.path.join(mask_input_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}")
            continue

        success = process_image_pair(image_path, mask_path)
        status = "Success" if success else "Failed"
        print(f"{status}: {filename}")