import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
import re

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
            area = np.sum(slice_data >= 0.99)  # Changed from == 1 to >= 0.99
            
            if area > max_area:
                max_area = area
                best_slice_idx = z
        
        print(f"  Best slice index (z): {best_slice_idx}")
        print(f"  Max area with value>=0.99: {max_area} pixels")
        
        if max_area == 0:
            print(f"  Warning: No pixels with value>=0.99 found")
            return [], best_slice_idx, None, (pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        
        # Get the best slice
        best_slice = data[:, :, best_slice_idx]
        
        # Find connected components to identify left and right regions
        labeled_array, num_features = ndimage.label(best_slice >= 0.99)
        
        # Calculate centers of mass for all regions
        regions_info = []
        centers = []
        
        for i in range(1, num_features + 1):
            region_mask = (labeled_array == i)
            center_of_mass = ndimage.center_of_mass(region_mask)
            
            # Convert center coordinates (row, col) to (x, y, z) in pixels
            center_x = center_of_mass[1]  # column -> x
            center_y = center_of_mass[0]  # row -> y
            center_z = best_slice_idx  # z coordinate is the slice index
            
            centers.append((center_x, center_y, center_z))
            regions_info.append({
                'center': (center_x, center_y, center_z),
                'corneal_anterior': None,
                'corneal_anterior_3d': None,
                'mask': region_mask
            })
        
        # Sort regions by x-coordinate to get consistent left-right ordering
        sorted_regions = sorted(regions_info, key=lambda r: r['center'][0])
        
        # Print 3D coordinates for left and right eye centers in pixels
        if len(sorted_regions) >= 2:
            left_eye = sorted_regions[0]
            right_eye = sorted_regions[-1]
            
            print("\nEye Center 3D Coordinates (in pixels):")
            print(f"Left Eye Center (x, y, z): ({left_eye['center'][0]:.1f}, {left_eye['center'][1]:.1f}, {left_eye['center'][2]})")
            print(f"Right Eye Center (x, y, z): ({right_eye['center'][0]:.1f}, {right_eye['center'][1]:.1f}, {right_eye['center'][2]})")
        else:
            print(f"  Warning: Found {len(sorted_regions)} regions, expected at least 2 for left and right eyes")
        
        # Calculate 3D corneal anterior points for each region
        print("\nCalculating 3D corneal anterior points...")
        
        # For each region in the best slice, find corresponding 3D region in the entire volume
        for i, region_info in enumerate(regions_info):
            center = region_info['center']
            center_x, center_y, center_z = center
            
            # Find all voxels in the 3D volume that belong to this eye
            # Use a region growing approach starting from the center
            eye_3d_coords = []
            
            # Get all coordinates where value >= 0.99 in the entire volume
            all_coords = np.where(data >= 0.99)
            all_points = list(zip(all_coords[1], all_coords[0], all_coords[2]))  # (x, y, z)
            
            if len(all_points) == 0:
                continue
            
            # For simplicity, assign each 3D point to the nearest 2D region center
            min_distance_to_center = float('inf')
            eye_3d_coords = []
            
            for x, y, z in all_points:
                # Calculate distance to this region's center (only considering x, y)
                distance_2d = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Check if this is the closest region center for this point
                is_closest = True
                for other_region in regions_info:
                    other_center = other_region['center']
                    other_distance_2d = np.sqrt((x - other_center[0])**2 + (y - other_center[1])**2)
                    if other_distance_2d < distance_2d:
                        is_closest = False
                        break
                
                if is_closest:
                    eye_3d_coords.append((x, y, z))
            
            if len(eye_3d_coords) == 0:
                continue
            
            # Find the point with maximum 3D distance from center
            max_distance_3d = 0
            farthest_point_3d = None
            
            for x, y, z in eye_3d_coords:
                distance_3d = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                if distance_3d > max_distance_3d:
                    max_distance_3d = distance_3d
                    farthest_point_3d = (x, y, z)
            
            # Store the 3D corneal anterior point
            regions_info[i]['corneal_anterior_3d'] = farthest_point_3d
            
            # Determine eye side
            eye_side = "Left" if i == 0 else "Right"
            
            print(f"  {eye_side} Eye:")
            print(f"    3D Corneal Anterior (farthest point): ({farthest_point_3d[0]:.1f}, {farthest_point_3d[1]:.1f}, {farthest_point_3d[2]})")
            print(f"    Distance from center (3D): {max_distance_3d:.2f} pixels")
        
        # Continue with the original 2D corneal anterior calculation for comparison
        if len(regions_info) >= 2:
            # Sort regions by x-coordinate to get consistent left-right ordering
            sorted_indices = sorted(range(len(regions_info)), key=lambda i: regions_info[i]['center'][0])
            
            # Calculate the line connecting the two main centers (leftmost and rightmost)
            left_center = regions_info[sorted_indices[0]]['center']
            right_center = regions_info[sorted_indices[-1]]['center']
            
            # Direction vector of the line connecting two centers
            dx = right_center[0] - left_center[0]
            dy = right_center[1] - left_center[1]
            
            # Calculate 2D corneal anterior points for each region
            print("\nCalculating 2D corneal anterior points for comparison...")
            for i in range(len(regions_info)):
                region_mask = regions_info[i]['mask']
                center = regions_info[i]['center']
                
                # Find all points in this region (2D slice)
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
                    
                    # Calculate 2D distance
                    corneal_2d = regions_info[i]['corneal_anterior']
                    distance_2d = np.sqrt((corneal_2d[0] - center[0])**2 + (corneal_2d[1] - center[1])**2)
                    
                    # Determine eye side
                    eye_side = "Left" if i == 0 else "Right"
                    
                    print(f"  {eye_side} Eye:")
                    print(f"    2D Corneal Anterior (angle-based): ({corneal_2d[0]:.1f}, {corneal_2d[1]:.1f}, {center[2]})")
                    print(f"    Distance from center (2D): {distance_2d:.2f} pixels")
        
        else:
            # For single region, fall back to original method (x < center_x)
            for i in range(len(regions_info)):
                region_mask = regions_info[i]['mask']
                center = regions_info[i]['center']
                center_x, center_y = center[0], center[1]
                
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

def find_matching_files(quantu_files, eyeball_files):
    """
    Find matching files based on numeric parts in filenames
    """
    quantu_dict = {}
    eyeball_dict = {}
    
    # Build dictionaries with number patterns as keys
    for file in quantu_files:
        numbers = extract_numbers_from_filename(file)
        if numbers:
            quantu_dict[numbers] = file
    
    for file in eyeball_files:
        numbers = extract_numbers_from_filename(file)
        if numbers:
            eyeball_dict[numbers] = file
    
    # Find common number patterns
    common_numbers = set(quantu_dict.keys()).intersection(set(eyeball_dict.keys()))
    
    # Return pairs of matching files
    matching_pairs = []
    for numbers in common_numbers:
        matching_pairs.append((quantu_dict[numbers], eyeball_dict[numbers]))
    
    return matching_pairs

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
    quantu_files = [f for f in os.listdir(quantu_directory) if f.endswith('.nii.gz')]
    eyeball_files = [f for f in os.listdir(eyeball_directory) if f.endswith('.nii.gz')]
    
    # Find matching files based on numeric parts
    matching_pairs = find_matching_files(quantu_files, eyeball_files)
    
    if not matching_pairs:
        print("No matching files found between the two directories based on numeric parts")
        return
    
    print(f"Found {len(matching_pairs)} matching file pairs to process")
    print("="*60)
    
    # Prepare data for CSV output
    csv_data = []
    zygomatic_distance_data = []
    corneal_distance_data = []  # New data for corneal anterior distances
    
    for quantu_file, eyeball_file in sorted(matching_pairs):
        print(f"Processing: {quantu_file} <-> {eyeball_file}")
        
        quantu_path = os.path.join(quantu_directory, quantu_file)
        eyeball_path = os.path.join(eyeball_directory, eyeball_file)
        
        # Get zygomatic positions from quantu file
        quantu_positions, quantu_voxel_sizes = get_quantu_positions(quantu_path)
        
        # Get eyeball positions from eyeball file
        eyeball_regions, best_slice_idx, best_slice, voxel_sizes = get_eyeball_positions(eyeball_path)
        
        # Extract voxel sizes for distance calculations (use eyeball file voxel sizes)
        pixel_spacing_x, pixel_spacing_y, pixel_spacing_z = voxel_sizes
        
        # Calculate zygomatic distance
        zygomatic_distance_mm = np.nan
        if len(quantu_positions) == 2:
            # Calculate distance between two zygomatic points in pixels
            pos1, pos2 = quantu_positions[0], quantu_positions[1]
            distance_pixels = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Convert to millimeters using average pixel spacing (assuming isotropic in x-y plane)
            # Use pixel_spacing_x for x-direction and pixel_spacing_y for y-direction
            distance_x_mm = (pos2[0] - pos1[0]) * pixel_spacing_x
            distance_y_mm = (pos2[1] - pos1[1]) * pixel_spacing_y
            zygomatic_distance_mm = np.sqrt(distance_x_mm**2 + distance_y_mm**2)
            
            print(f"  Zygomatic distance: {zygomatic_distance_mm:.2f} mm ({distance_pixels:.1f} pixels)")
        else:
            print(f"  Warning: Found {len(quantu_positions)} zygomatic points, expected 2")
        
        # Add zygomatic distance data
        zygomatic_distance_data.append({
            'quantu_filename': quantu_file,
            'eyeball_filename': eyeball_file,
            'zygomatic_distance_mm': zygomatic_distance_mm
        })
        
        # Collect corneal anterior distance data
        corneal_row = {
            'quantu_filename': quantu_file,
            'eyeball_filename': eyeball_file,
            'left_eye_2d_corneal_distance_pixels': np.nan,
            'right_eye_2d_corneal_distance_pixels': np.nan,
            'left_eye_3d_corneal_distance_pixels': np.nan,
            'right_eye_3d_corneal_distance_pixels': np.nan
        }
        
        if best_slice is None:
            print(f"  No valid slice found in eyeball file: {eyeball_file}")
            # Add row with NaN values for failed cases
            csv_data.append({
                'quantu_filename': quantu_file,
                'eyeball_filename': eyeball_file,
                'left_eye_exophthalmos_mm': np.nan,
                'right_eye_exophthalmos_mm': np.nan
            })
            corneal_distance_data.append(corneal_row)
            continue
        
        # Extract corneal anterior distances from eyeball_regions
        if eyeball_regions and len(eyeball_regions) >= 2:
            # Sort regions by x-coordinate to ensure consistent left-right ordering
            sorted_regions = sorted(eyeball_regions, key=lambda r: r['center'][0])
            
            # Left eye (first region)
            left_region = sorted_regions[0]
            if left_region['corneal_anterior']:
                left_2d_distance = np.sqrt(
                    (left_region['corneal_anterior'][0] - left_region['center'][0])**2 + 
                    (left_region['corneal_anterior'][1] - left_region['center'][1])**2
                )
                corneal_row['left_eye_2d_corneal_distance_pixels'] = left_2d_distance
            
            if left_region['corneal_anterior_3d']:
                left_3d_distance = np.sqrt(
                    (left_region['corneal_anterior_3d'][0] - left_region['center'][0])**2 + 
                    (left_region['corneal_anterior_3d'][1] - left_region['center'][1])**2 + 
                    (left_region['corneal_anterior_3d'][2] - left_region['center'][2])**2
                )
                corneal_row['left_eye_3d_corneal_distance_pixels'] = left_3d_distance
            
            # Right eye (last region)
            right_region = sorted_regions[-1]
            if right_region['corneal_anterior']:
                right_2d_distance = np.sqrt(
                    (right_region['corneal_anterior'][0] - right_region['center'][0])**2 + 
                    (right_region['corneal_anterior'][1] - right_region['center'][1])**2
                )
                corneal_row['right_eye_2d_corneal_distance_pixels'] = right_2d_distance
            
            if right_region['corneal_anterior_3d']:
                right_3d_distance = np.sqrt(
                    (right_region['corneal_anterior_3d'][0] - right_region['center'][0])**2 + 
                    (right_region['corneal_anterior_3d'][1] - right_region['center'][1])**2 + 
                    (right_region['corneal_anterior_3d'][2] - right_region['center'][2])**2
                )
                corneal_row['right_eye_3d_corneal_distance_pixels'] = right_3d_distance
        
        # Add corneal distance data
        corneal_distance_data.append(corneal_row)
        
        # Calculate exophthalmos
        print(f"  Calculating exophthalmos...")
        exophthalmos_results = calculate_exophthalmos(eyeball_regions, quantu_positions, voxel_sizes)
        
        # Prepare CSV row data
        row_data = {
            'quantu_filename': quantu_file,
            'eyeball_filename': eyeball_file
        }
        
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
        
        print(f"  Processed {quantu_file} <-> {eyeball_file} successfully")
        if exophthalmos_results:
            for result in exophthalmos_results:
                print(f"    {result['eye_side']} Eye: {result['total_exophthalmos_mm']:.2f} mm")
        print("-" * 40)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    output_file = os.path.join(output_directory, '3d_exophthalmos_results.csv')
    df.to_csv(output_file, index=False)
    
    # Create DataFrame for zygomatic distance and save to CSV
    df_zygomatic = pd.DataFrame(zygomatic_distance_data)
    zygomatic_output_file = os.path.join(output_directory, '3d_zygomatic_distance_results.csv')
    df_zygomatic.to_csv(zygomatic_output_file, index=False)
    
    # Create DataFrame for corneal anterior distances and save to CSV
    df_corneal = pd.DataFrame(corneal_distance_data)
    corneal_output_file = os.path.join(output_directory, '3d_corneal_anterior_distances.csv')
    df_corneal.to_csv(corneal_output_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Exophthalmos: {output_file}")
    print(f"  Zygomatic distance: {zygomatic_output_file}")
    print(f"  Corneal anterior distances: {corneal_output_file}")
    print(f"Total files processed: {len(csv_data)}")
    print("\nExophthalmos Summary:")
    print(df.describe())
    print("\nZygomatic Distance Summary:")
    print(df_zygomatic.describe())
    print("\nCorneal Anterior Distances Summary:")
    print(df_corneal.describe())

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
    quantu_directory = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\3DMRI\Quantu\计算用\mask"
    eyeball_directory = r"C:\Users\xrVis001\Desktop\finaldata0528\finaldata0528\3DMRI\Eyeball\计算用\mask"
    output_directory = r"C:\Users\xrVis001\Desktop\finaldata0528"
    
    # Process corresponding files and generate CSV output
    print("=== PROCESSING FILES AND GENERATING CSV ===")
    print("Processing corresponding files to calculate exophthalmos and generate CSV output...")
    print(f"Quantu directory: {quantu_directory}")
    print(f"Eyeball directory: {eyeball_directory}")
    print(f"Output directory: {output_directory}")
    print("="*60)
    
    process_corresponding_files(quantu_directory, eyeball_directory, output_directory)
