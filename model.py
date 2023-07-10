import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 padding=2,
                 dilation=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            )
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = 1

    def forward(self, x):
        return self.double_conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.down_sample = nn.MaxPool2d(kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return self.down_sample(x)


class Bridge(nn.Module):
    def __init__(self, in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 padding=2,
                 dilation=2):
        super().__init__()
        self.bridge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = 1

    def forward(self, x):
        return self.bridge_conv(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch = nn.BatchNorm2d(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        x1 = self.batch(x1)
        x1 = self.activation(x1)
        cat = torch.cat([x1, x2], dim=1)
        return cat


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class UNet(nn.Module):
    def __init__(self, filter_size, n_channels, n_classes):
        super().__init__()
        self.filter_size = filter_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Block Down 1
        self.conv_block1 = DoubleConvBlock(n_channels, filter_size, filter_size)
        self.down_sample_block1 = DownSampleBlock()

        # Block Down 2
        self.conv_block2 = DoubleConvBlock(filter_size, filter_size*2, filter_size*2)
        self.down_sample_block2 = DownSampleBlock()

        # Block Down 3
        self.conv_block3 = DoubleConvBlock(filter_size*2, filter_size*3, filter_size*3)
        self.down_sample_block3 = DownSampleBlock()

        # Bridge Block
        self.bridge = Bridge(filter_size*3, filter_size*4, filter_size*4)

        # Up Sample Block 1
        self.up_sample_block1 = UpSampleBlock(filter_size*4, filter_size*4)
        self.conv_block4 = DoubleConvBlock(filter_size*4+filter_size*3, filter_size*3, filter_size*3)

        # Up Sample Block 2
        self.up_sample_block2 = UpSampleBlock(filter_size*3, filter_size*3)
        self.conv_block5 = DoubleConvBlock(filter_size*3+filter_size*2, filter_size*2, filter_size*2)

        # Up Sample Block 3
        self.up_sample_block3 = UpSampleBlock(filter_size*2, filter_size*2)
        self.conv_block6 = DoubleConvBlock(filter_size*2+n_channels, filter_size, filter_size)

        # Output Block
        self.output = OutConv(filter_size, n_classes)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.down_sample_block1(x1)
        x3 = self.conv_block2(x2)
        x4 = self.down_sample_block1(x3)
        x5 = self.conv_block3(x4)
        x6 = self.down_sample_block1(x5)
        x7 = self.bridge(x6)
        x8 = self.up_sample_block1(x7, x5)
        x9 = self.conv_block4(x8)
        x10 = self.up_sample_block2(x9, x3)
        x11 = self.conv_block5(x10)
        x12 = self.up_sample_block3(x11, x)
        x13 = self.conv_block6(x12)
        x14 = self.output(x13)
        return x14

    def print_layers(self, dim):
        x = torch.rand(1, 3, dim[0], dim[1])
        print()
        print("####### Input #######")
        print("Tensor Input: {}".format(x.shape))
        print()

        print("###### Down Block 1 #######")
        x1 = self.conv_block1(x)
        print("Output Conv Block 1: {}".format(x1.shape))
        h, w = self.calculate_size(x.shape[2],
                                   x.shape[3],
                                   self.conv_block1.padding,
                                   self.conv_block1.dilation,
                                   self.conv_block1.kernel_size,
                                   self.conv_block1.stride)
        print("Calculate Conv Block 1: {}".format([1, self.filter_size, h, w]))
        x2 = self.down_sample_block1(x1)
        print("Output Down Sample Block 1: {}".format(x2.shape))
        h, w = self.calculate_size(x1.shape[2],
                                   x1.shape[3],
                                   0,
                                   0,
                                   self.down_sample_block1.kernel_size,
                                   self.down_sample_block1.stride)
        print("Calculate Down Block 1: {}".format([1, self.filter_size, h, w]))
        print()

        print("###### Down Block 2 #######")
        x3 = self.conv_block2(x2)
        print("Output Conv Block 2: {}".format(x3.shape))
        h, w = self.calculate_size(x2.shape[2],
                                   x2.shape[3],
                                   self.conv_block2.padding,
                                   self.conv_block2.dilation,
                                   self.conv_block2.kernel_size,
                                   self.conv_block2.stride)
        print("Calculate Conv Block 2: {}".format([1, 2*self.filter_size, h, w]))
        x4 = self.down_sample_block1(x3)
        print("Output Down Sample Block 2: {}".format(x4.shape))
        h, w = self.calculate_size(x3.shape[2],
                                   x3.shape[3],
                                   0,
                                   0,
                                   self.down_sample_block2.kernel_size,
                                   self.down_sample_block2.stride)
        print("Calculate Down Block 2: {}".format([1, 2*self.filter_size, h, w]))
        print()

        print("###### Down Block 3 #######")
        x5 = self.conv_block3(x4)
        print("Output Conv Block 3: {}".format(x5.shape))
        h, w = self.calculate_size(x4.shape[2],
                                   x4.shape[3],
                                   self.conv_block3.padding,
                                   self.conv_block3.dilation,
                                   self.conv_block3.kernel_size,
                                   self.conv_block3.stride)
        print("Calculate Conv Block 3: {}".format([1, 3 * self.filter_size, h, w]))
        x6 = self.down_sample_block1(x5)
        print("Output Down Sample Block 3: {}".format(x6.shape))
        h, w = self.calculate_size(x5.shape[2],
                                   x5.shape[3],
                                   0,
                                   0,
                                   self.down_sample_block3.kernel_size,
                                   self.down_sample_block3.stride)
        print("Calculate Down Block 3: {}".format([1, 3 * self.filter_size, h, w]))
        print()

        print("###### Bridge #######")
        x7 = self.bridge(x6)
        print("Output Conv Bridge: {}".format(x7.shape))
        h, w = self.calculate_size(x6.shape[2],
                                   x6.shape[3],
                                   self.bridge.padding,
                                   self.bridge.dilation,
                                   self.bridge.kernel_size,
                                   self.bridge.stride)
        print("Calculate Conv Bridge: {}".format([1, 4 * self.filter_size, h, w]))
        print()

        print("###### Up Block 4 #######")
        x8 = self.up_sample_block1(x7, x5)
        print("Output Down Sample Block 4: {}".format(x8.shape))
        x9 = self.conv_block4(x8)
        print("Output Conv Block 4: {}".format(x9.shape))
        print()

        print("###### Up Block 5 #######")
        x10 = self.up_sample_block2(x9, x3)
        print("Output Down Sample Block 5: {}".format(x10.shape))
        x11 = self.conv_block5(x10)
        print("Output Conv Block 5: {}".format(x11.shape))
        print()

        print("###### Up Block 6 #######")
        x12 = self.up_sample_block3(x11, x)
        print("Output Down Sample Block 6: {}".format(x12.shape))
        x13 = self.conv_block6(x12)
        print("Output Conv Block 6: {}".format(x13.shape))
        print()

        print("###### Final Conv #######")
        x14 = self.output(x13)
        print("Output Block: {}".format(x14.shape))
        print()

    @staticmethod
    def calculate_size(h, w, padding, dilation, kernel_size, stride):
        height = (h + 2 * padding - dilation * (kernel_size - 1)) / stride
        width = (w + 2 * padding - dilation * (kernel_size - 1)) / stride
        return height, width

