# Exophthalmos-based-on-MR
Including identification and eyeball segmentation

## 1.MR序列
- 适配2D、3D 眼眶MR横断位序列
- 训练用数据集（来源Prisma 3T MR）：
  - 2D：T2 DIXON 横断位 同相位数据
  - 3D： T2WI
 

## 2.图像预处理部分
（1）图像标准化处理 —— 对应代码为xxx
（2）图像裁剪 （320，112，80 / 320，112，20）

## 3.颧突顶点识别


## 4.眼球分割——UNET
（1）2D
（2）3D

## 5.突眼度计算


## 6.基于Web-APP的整合？
