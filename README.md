# Exophthalmos Measurement Based on MR Images  
Including landmark identification and eyeball segmentation  

## 1. MR Sequences  
- Supports 2D and 3D orbital MR transverse sequences  
- Training dataset (from Prisma 3T MR):  
  - **2D**: T2 DIXON transverse in-phase images  
  - **3D**: T2-weighted imaging (T2WI)  

## 2. Image Preprocessing  
(1) **Image Standardization** – Implemented in [Standardize.py](file:///Users/owme/Documents/GitHub/Exophthalmos-based-on-MR/01%20Preprocessing/Standardize.py)  
(2) **Image Cropping** (dimensions: 320×112×80 / 320×112×20) – Implemented in [Crop.py](file:///Users/owme/Documents/GitHub/Exophthalmos-based-on-MR/01%20Preprocessing/Crop.py)  

## 3. Zygomatic Process Vertex Identification  wcy
(1) 2D 
(2) 3D   

## 4. Eyeball Segmentation using UNET  
(1) 2D implementation  
(2) 3D implementation  

## 5. Exophthalmos Measurement Calculation   wcy

## 6. Web Application Integration 
*(Concept under consideration)*  
