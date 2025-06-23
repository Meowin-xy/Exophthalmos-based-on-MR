import os
import numpy as np
import nibabel as nib

# Set source and target directories
source_dir = r'G:\2506_3Dtranifti'
target_dir =r'G:\2506_3Dtraniftistandardize'

def z_score_normalize(data):
    """Perform Z-score normalization"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1e-6  # Avoid division by zero
    return (data - mean) / std

def process_nifti_files():
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Traverse all subdirectories and files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                source_path = os.path.join(root, file)
                
                # Construct target path (preserving directory structure)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                try:
                    # Load NIfTI file
                    img = nib.load(source_path)
                    data = img.get_fdata()
                    
                    # Convert to float and normalize
                    normalized_data = z_score_normalize(data.astype(np.float32))
                    
                    # Create new NIfTI image (preserving original spatial information)
                    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
                    normalized_img.header.set_data_dtype(np.float32)
                    
                    # Save processed file
                    nib.save(normalized_img, target_path)
                    print(f"Processed successfully: {source_path} -> {target_path}")
                
                except Exception as e:
                    print(f"Processing failed: {source_path} | Error: {str(e)}")

if __name__ == "__main__":
    process_nifti_files()