import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.cnn_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
            bias=True,
        )
        self.act = nn.ReLU()
        self.cnn_2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
            bias=True,
        )


    def forward(self, x):
        return self.cnn_2(self.act(self.cnn_1(x))


class AttnConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=1)
        self.cnn_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
            bias=True,
        )
        self.act = nn.ReLU()
        self.cnn_2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        init_x = x
        x = self.pool(x)
        x = self.act(self.cnn_1(x))
        x = self.cnn_2(x)
        x = self.sigmoid(x)
        return x + init_x

class Cell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_cells=3):
        super().__init__()
        self.num_cells = num_cells
        self.conv_block = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size
        )
        self.attn_block = AttnBlock(
            in_channels, 
            out_channels, 
            kernel_size
        )

    def forward(self, x):
        init_x = x
        for _ in self.num_cells:
            prev_x = x
            x = self.conv_block(x)
            x = self.attn_block(x)
            x = prev_x + x
        return init_x + x


class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, scale):
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=3,
        )
        self.pxl_shuffle = nn.PixelShuffle(scale)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=3,
            kernel_size=1
        )

    def forward(self, x)
        return self.conv2(self.pxl_shuffle(self.conv1(x)))
