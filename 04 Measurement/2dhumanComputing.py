import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

def find_points_in_nii_files(directory_path):
    """
    Read nii.gz files from specified directory and find coordinates of points with value 1
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    # Get all nii.gz files in the directory
    nii_files = [f for f in os.listdir(directory_path) if f.endswith('.nii.gz')]
    
    if not nii_files:
        print(f"No nii.gz files found in directory: {directory_path}")
        return
    
    print(f"Found {len(nii_files)} nii.gz files")
    print("="*60)
    
    results = {}
    
    for file_name in nii_files:
        file_path = os.path.join(directory_path, file_name)
        
        try:
            # Load nii.gz file
            img = nib.load(file_path)
            data = img.get_fdata()
            
            print(f"File: {file_name}")
            print(f"Data shape: {data.shape}")
            
            # Find coordinates where value equals 1
            coords = np.where(data == 1)
            
            if len(coords[0]) == 0:
                print("  No points with value 1 found")
            else:
                print(f"  Found {len(coords[0])} points with value 1:")
                
                # Store coordinates for this file
                point_coords = []
                
                for i in range(len(coords[0])):
                    if len(coords) == 3:  # 3D data
                        coord = (coords[0][i], coords[1][i], coords[2][i])
                        print(f"    Point {i+1}: (x={coords[1][i]}, y={coords[0][i]}, z={coords[2][i]})")
                    else:  # 2D data
                        coord = (coords[0][i], coords[1][i])
                        print(f"    Point {i+1}: (x={coords[1][i]}, y={coords[0][i]})")
                    
                    point_coords.append(coord)
                
                results[file_name] = point_coords
            
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            print("-" * 40)
    
    return results

def get_quantu_positions(file_path):
    """
    Get zygomatic (quantu) positions from a single nii.gz file
    """
    try:
        # Load nii.gz file
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Find coordinates where value equals 1
        coords = np.where(data == 1)
        
        if len(coords[0]) == 0:
            return []
        
        # Store coordinates for this file
        point_coords = []
        
        for i in range(len(coords[0])):
            if len(coords) == 3:  # 3D data
                coord = (coords[1][i], coords[0][i], coords[2][i])  # (x, y, z)
            else:  # 2D data
                coord = (coords[1][i], coords[0][i])  # (x, y)
            
            point_coords.append(coord)
        
        return point_coords
        
    except Exception as e:
        print(f"Error processing quantu file {file_path}: {str(e)}")
        return []

def get_eyeball_positions(file_path):
    """
    Get center of mass and corneal anterior positions from eyeball nii.gz file
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
        
        print(f"  Voxel sizes: x={pixel_spacing_x:.3f}mm, y={pixel_spacing_y:.3f}mm, z={pixel_spacing_z:.3f}mm")
        
        # Find the slice with maximum area of value 1
        max_area = 0
        best_slice_idx = 0
        
        for z in range(data.shape[2]):
            slice_data = data[:, :, z]
            area = np.sum(slice_data == 1)
            
            if area > max_area:
                max_area = area
                best_slice_idx = z
        
        if max_area == 0:
            return [], best_slice_idx, None, (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
        # Get the best slice
        best_slice = data[:, :, best_slice_idx]
        
        # Find connected components to identify left and right regions
        labeled_array, num_features = ndimage.label(best_slice == 1)
        
        # First pass: calculate centers of mass for all regions
        regions_info = []
        centers = []
        
        for i in range(1, num_features + 1):
            region_mask = (labeled_array == i)
            center_of_mass = ndimage.center_of_mass(region_mask)
            
            # Convert center coordinates (row, col) to (x, y)
            center_x = center_of_mass[1]  # column -> x
            center_y = center_of_mass[0]  # row -> y
            
            centers.append((center_x, center_y))
            regions_info.append({
                'center': (center_x, center_y),
                'corneal_anterior': None,
                'mask': region_mask
            })
        
        # If we have at least 2 regions, calculate corneal anterior points using new algorithm
        if len(regions_info) >= 2:
            # Sort regions by x-coordinate to get consistent left-right ordering
            sorted_indices = sorted(range(len(regions_info)), key=lambda i: regions_info[i]['center'][0])
            
            # Calculate the line connecting the two main centers (leftmost and rightmost)
            left_center = regions_info[sorted_indices[0]]['center']
            right_center = regions_info[sorted_indices[-1]]['center']
            
            # Direction vector of the line connecting two centers
            dx = right_center[0] - left_center[0]
            dy = right_center[1] - left_center[1]
            
            # Calculate corneal anterior points for each region
            for i in range(len(regions_info)):
                region_mask = regions_info[i]['mask']
                center = regions_info[i]['center']
                
                # Find all points in this region
                region_coords = np.where(region_mask)
                region_y_coords = region_coords[0]  # row coordinates
                region_x_coords = region_coords[1]  # column coordinates
                
                # Find corneal anterior point: within 45-135 degree range and maximum distance from center
                anterior_candidates = []
                
                for j in range(len(region_x_coords)):
                    x, y = region_x_coords[j], region_y_coords[j]
                    
                    # Additional condition: x coordinate must be less than center x
                    if x < center[0]:
                        # Vector from center to current point
                        point_dx = x - center[0]
                        point_dy = y - center[1]
                        
                        # Calculate angle between center line and point vector
                        if dx != 0 or dy != 0:  # Avoid division by zero
                            # Use dot product to calculate angle
                            dot_product = dx * point_dx + dy * point_dy
                            center_line_length = np.sqrt(dx*dx + dy*dy)
                            point_vector_length = np.sqrt(point_dx*point_dx + point_dy*point_dy)
                            
                            if center_line_length > 0 and point_vector_length > 0:
                                cos_angle = dot_product / (center_line_length * point_vector_length)
                                # Clamp cos_angle to [-1, 1] to avoid numerical errors
                                cos_angle = max(-1, min(1, cos_angle))
                                angle_rad = np.arccos(cos_angle)
                                angle_deg = np.degrees(angle_rad)
                                
                                # Check if angle is within 45-135 degree range
                                if 45 <= angle_deg <= 135:
                                    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                                    anterior_candidates.append((x, y, distance))
                
                # Find the point with maximum distance within the angle range
                if anterior_candidates:
                    anterior_candidates.sort(key=lambda p: p[2], reverse=True)
                    regions_info[i]['corneal_anterior'] = (anterior_candidates[0][0], anterior_candidates[0][1])
        
        else:
            # For single region, fall back to original method (x < center_x)
            for i in range(len(regions_info)):
                region_mask = regions_info[i]['mask']
                center = regions_info[i]['center']
                center_x, center_y = center
                
                # Find all points in this region
                region_coords = np.where(region_mask)
                region_y_coords = region_coords[0]  # row coordinates
                region_x_coords = region_coords[1]  # column coordinates
                
                # Original method: x < center_x and maximum distance from center
                anterior_candidates = []
                for j in range(len(region_x_coords)):
                    x, y = region_x_coords[j], region_y_coords[j]
                    if x < center_x:  # x coordinate less than center
                        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        anterior_candidates.append((x, y, distance))
                
                # Find the point with maximum distance
                if anterior_candidates:
                    anterior_candidates.sort(key=lambda p: p[2], reverse=True)
                    regions_info[i]['corneal_anterior'] = (anterior_candidates[0][0], anterior_candidates[0][1])
        
        return regions_info, best_slice_idx, best_slice, (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
    except Exception as e:
        print(f"Error processing eyeball file {file_path}: {str(e)}")
        return [], 0, None, (1.0, 1.0, 1.0)

def process_corresponding_files(quantu_directory, eyeball_directory, output_directory):
    """
    Process corresponding files from both directories and output CSV with exophthalmos results
    """
    # Check if directories exist
    if not os.path.exists(quantu_directory):
        print(f"Quantu directory does not exist: {quantu_directory}")
        return
    
    if not os.path.exists(eyeball_directory):
        print(f"Eyeball directory does not exist: {eyeball_directory}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    # Get all nii.gz files from both directories
    quantu_files = set([f for f in os.listdir(quantu_directory) if f.endswith('.nii.gz')])
    eyeball_files = set([f for f in os.listdir(eyeball_directory) if f.endswith('.nii.gz')])
    
    # Find common files
    common_files = quantu_files.intersection(eyeball_files)
    
    if not common_files:
        print("No common files found between the two directories")
        return
    
    print(f"Found {len(common_files)} common files to process")
    print("="*60)
    
    # Prepare data for CSV output
    csv_data = []
    
    for file_name in sorted(common_files):
        print(f"Processing: {file_name}")
        
        quantu_path = os.path.join(quantu_directory, file_name)
        eyeball_path = os.path.join(eyeball_directory, file_name)
        
        # Get zygomatic positions from quantu file
        quantu_positions = get_quantu_positions(quantu_path)
        
        # Get eyeball positions from eyeball file
        eyeball_regions, best_slice_idx, best_slice, voxel_sizes = get_eyeball_positions(eyeball_path)
        
        if best_slice is None:
            print(f"  No valid slice found in eyeball file: {file_name}")
            # Add row with NaN values for failed cases
            csv_data.append({
                'filename': file_name,
                'left_eye_exophthalmos_mm': np.nan,
                'right_eye_exophthalmos_mm': np.nan
            })
            continue
        
        # Calculate exophthalmos
        print(f"  Calculating exophthalmos...")
        exophthalmos_results = calculate_exophthalmos(eyeball_regions, quantu_positions, voxel_sizes)
        
        # Prepare CSV row data
        row_data = {'filename': file_name}
        
        if exophthalmos_results:
            # Sort results by eye side to ensure consistent ordering
            left_eye_result = None
            right_eye_result = None
            
            for result in exophthalmos_results:
                if result['eye_side'] == 'Left':
                    left_eye_result = result
                elif result['eye_side'] == 'Right':
                    right_eye_result = result
            
            # Add exophthalmos values
            row_data['left_eye_exophthalmos_mm'] = left_eye_result['total_exophthalmos_mm'] if left_eye_result else np.nan
            row_data['right_eye_exophthalmos_mm'] = right_eye_result['total_exophthalmos_mm'] if right_eye_result else np.nan
        else:
            row_data['left_eye_exophthalmos_mm'] = np.nan
            row_data['right_eye_exophthalmos_mm'] = np.nan
        
        csv_data.append(row_data)
        
        print(f"  Processed {file_name} successfully")
        if exophthalmos_results:
            for result in exophthalmos_results:
                print(f"    {result['eye_side']} Eye: {result['total_exophthalmos_mm']:.2f} mm")
        print("-" * 40)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    output_file = os.path.join(output_directory, 'exophthalmos_results.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total files processed: {len(csv_data)}")
    print("\nSummary:")
    print(df.describe())

def calculate_exophthalmos(eyeball_regions, quantu_positions, voxel_sizes):
    """
    Calculate exophthalmos (突眼度) for each eye
    左眼突眼度 = 左眼球中心到两个颧突连线的距离 + 左眼角膜前缘到左眼球中心的距离
    右眼突眼度 = 右眼球中心到两个颧突连线的距离 + 右眼角膜前缘到右眼球中心的距离
    All distances are calculated in millimeters using voxel size information
    """
    exophthalmos_results = []
    
    # Extract voxel sizes
    pixel_spacing_x, pixel_spacing_y, pixel_spacing_z = voxel_sizes
    
    # Check if we have exactly 2 zygomatic points
    if len(quantu_positions) != 2:
        print(f"  Warning: Expected 2 zygomatic points, found {len(quantu_positions)}")
        return exophthalmos_results
    
    # Sort zygomatic points by x-coordinate (left to right)
    sorted_quantu = sorted(quantu_positions, key=lambda p: p[0])
    left_quantu = sorted_quantu[0]
    right_quantu = sorted_quantu[1]
    
    # Convert zygomatic points to mm
    left_quantu_mm = (left_quantu[0] * pixel_spacing_x, left_quantu[1] * pixel_spacing_y)
    right_quantu_mm = (right_quantu[0] * pixel_spacing_x, right_quantu[1] * pixel_spacing_y)
    
    print(f"  Zygomatic points (mm): Left=({left_quantu_mm[0]:.2f}, {left_quantu_mm[1]:.2f}), Right=({right_quantu_mm[0]:.2f}, {right_quantu_mm[1]:.2f})")
    
    # Sort eyeball regions by x-coordinate (left to right)
    sorted_regions = sorted(eyeball_regions, key=lambda r: r['center'][0])
    
    for i, region in enumerate(sorted_regions):
        eye_side = "Left" if i == 0 else "Right"
        center = region['center']
        corneal_anterior = region['corneal_anterior']
        
        # Convert center to mm
        center_mm = (center[0] * pixel_spacing_x, center[1] * pixel_spacing_y)
        
        # Calculate distance from eyeball center to zygomatic line (in mm)
        # Line equation: from left_quantu_mm to right_quantu_mm
        # Distance from point to line formula: |ax + by + c| / sqrt(a² + b²)
        
        # Vector from left_quantu_mm to right_quantu_mm
        dx = right_quantu_mm[0] - left_quantu_mm[0]
        dy = right_quantu_mm[1] - left_quantu_mm[1]
        
        # Line equation: (y - y1) = (dy/dx) * (x - x1)
        # Rearranged to: dy*x - dx*y + (dx*y1 - dy*x1) = 0
        # So: a = dy, b = -dx, c = dx*y1 - dy*x1
        a = dy
        b = -dx
        c = dx * left_quantu_mm[1] - dy * left_quantu_mm[0]
        
        # Distance from eyeball center to zygomatic line (in mm)
        if abs(a) + abs(b) > 0:  # Avoid division by zero
            center_to_line_distance_mm = abs(a * center_mm[0] + b * center_mm[1] + c) / np.sqrt(a*a + b*b)
        else:
            center_to_line_distance_mm = 0
            print(f"    Warning: Cannot calculate distance to zygomatic line (points are identical)")
        
        # Calculate distance from corneal anterior to eyeball center (in mm)
        anterior_to_center_distance_mm = 0
        corneal_anterior_mm = None
        if corneal_anterior:
            corneal_anterior_mm = (corneal_anterior[0] * pixel_spacing_x, corneal_anterior[1] * pixel_spacing_y)
            anterior_to_center_distance_mm = np.sqrt(
                (corneal_anterior_mm[0] - center_mm[0])**2 + 
                (corneal_anterior_mm[1] - center_mm[1])**2
            )
        else:
            print(f"    Warning: No corneal anterior point found for {eye_side} eye")
        
        # Total exophthalmos (in mm)
        total_exophthalmos_mm = center_to_line_distance_mm + anterior_to_center_distance_mm
        
        result = {
            'eye_side': eye_side,
            'center': center,
            'center_mm': center_mm,
            'corneal_anterior': corneal_anterior,
            'corneal_anterior_mm': corneal_anterior_mm,
            'center_to_line_distance_pixels': abs(a * center[0] + b * center[1] + c) / np.sqrt(a*a + b*b) if abs(a) + abs(b) > 0 else 0,
            'center_to_line_distance_mm': center_to_line_distance_mm,
            'anterior_to_center_distance_pixels': np.sqrt((corneal_anterior[0] - center[0])**2 + (corneal_anterior[1] - center[1])**2) if corneal_anterior else 0,
            'anterior_to_center_distance_mm': anterior_to_center_distance_mm,
            'total_exophthalmos_mm': total_exophthalmos_mm
        }
        
        exophthalmos_results.append(result)
        
        print(f"    {eye_side} Eye Exophthalmos:")
        print(f"      Center to zygomatic line: {center_to_line_distance_mm:.2f} mm ({result['center_to_line_distance_pixels']:.1f} pixels)")
        print(f"      Anterior to center: {anterior_to_center_distance_mm:.2f} mm ({result['anterior_to_center_distance_pixels']:.1f} pixels)")
        print(f"      Total exophthalmos: {total_exophthalmos_mm:.2f} mm")
    
    return exophthalmos_results

# Main execution
if __name__ == "__main__":
    # Define directories
    quantu_directory = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\2DMRI\Quantu\quantuMaskmea"
    eyeball_directory = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\2DMRI\Eyeball\计算用\Mask2Dmea"
    output_directory = r"C:\Users\xrVis001\Desktop\finaldata0528"
    
    # Process corresponding files and generate CSV output
    print("=== PROCESSING FILES AND GENERATING CSV ===")
    print("Processing corresponding files to calculate exophthalmos and generate CSV output...")
    print(f"Quantu directory: {quantu_directory}")
    print(f"Eyeball directory: {eyeball_directory}")
    print(f"Output directory: {output_directory}")
    print("="*60)
    
    process_corresponding_files(quantu_directory, eyeball_directory, output_directory)
