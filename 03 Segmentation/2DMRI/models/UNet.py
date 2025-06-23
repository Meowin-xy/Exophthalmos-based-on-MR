import torch
from torch import nn
from torch.nn import functional as F

class BasicC3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicC3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x

class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        # Encoder (Downsampling Path)
        self.block_1_1_left = BasicC3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicC3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 1/2

        self.block_2_1_left = BasicC3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicC3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 1/4

        self.block_3_1_left = BasicC3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicC3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 1/8

        self.block_4_1_left = BasicC3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicC3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        # Decoder (Upsampling Path) with adjusted output_padding
        self.upconv_3 = nn.ConvTranspose3d(
            8 * n_filters, 
            4 * n_filters,
            kernel_size=2,
            stride=2,
            output_padding= (1,0,0)
        )
        self.block_3_1_right = BasicC3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicC3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(
            4 * n_filters, 
            2 * n_filters,
            kernel_size=2, 
            stride=2,output_padding=0
        )
        self.block_2_1_right = BasicC3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicC3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(
            2 * n_filters, 
            n_filters,
            kernel_size=2, 
            stride=2,output_padding=0
        )
        self.block_1_1_right = BasicC3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicC3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        # print("ds0:", ds0.shape)
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        # print("ds1:", ds1.shape)
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        # print("ds2:", ds2.shape)
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        # print("After pool3:", x.shape)

        # Decoder with skip connections
        x = self.upconv_3(x)
        x = x[:, :, :ds2.size(2), :, :]  
        x = torch.cat([x, ds2], dim=1)  # 现在尺寸应该完全匹配
        x = self.block_3_2_right(self.block_3_1_right(x))

        x = self.upconv_2(x)
        x = torch.cat([x, ds1], dim=1)
        x = self.block_2_2_right(self.block_2_1_right(x))

        x = self.upconv_1(x)
        x = torch.cat([x, ds0], dim=1)
        x = self.block_1_2_right(self.block_1_1_right(x))

        x = self.conv1x1(x)

        return torch.sigmoid(x) if self.n_cls == 1 else F.softmax(x, dim=1)