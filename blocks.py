import torch.nn as nn


#################################################################################################################################
############################################ Convolution-Related Blocks #########################################################
#################################################################################################################################
class Single_Conv_Block_2d(nn.Module):
    """Single Convolution Block for different parts of a UNet. This one is needed for example for the final block-part of a UNet
        because it does not contain any Max-Pooling- or Transpose-Operations.

        Original ConvBlock would have the following parameters:
            conv_kernel_size = 3
            conv_padding_size = 0
        But this would result in different input and output shapes.

    Args:
        -
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        with_relu_batchnorm=True,
        conv_kernel_size=3,
        conv_padding_size=1,
    ):
        """
        Args:
            in_channels (int): Input feature channels for the convolution
            out_channels (int): Output feature channels of the convolution
            with_relu_batchnorm (bool, optional): Applies batchnorm2d and relu after the convolution. Defaults to True.
            conv_kernel_size (int, optional): Kernel size for the convolution. Defaults to 3.
            conv_padding_size (int, optional): Padding size for the convolution. Defaults to 1.
        """
        super(Single_Conv_Block_2d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        if with_relu_batchnorm == False:
            self.conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding_size,
                ),
            )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class Conv_Block_2d(nn.Module):
    """Convolution Block for the contractive part of a UNet.

        Original ConvBlock would have the following parameters:
            conv_kernel_size = 3
            conv_padding_size = 0
            pool_kernel_size = 2
            pool_stride_size = 2
        But this would result in different input and output shapes.

    Args:
        -
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        conv_padding_size=1,
        pool_kernel_size=2,
        pool_stride_size=2,
    ):
        """
        Args:
            in_channels (int): Input feature channels for the convolution.
            out_channels (int): Output feature channels of the convolution.
            conv_kernel_size (int, optional): Kernel size for the convolution. Defaults to 3.
            conv_padding_size (int, optional): Padding size for the convolution. Defaults to 1.
            pool_kernel_size (int, optional): Kernel size for the pooling. Defaults to 2.
            pool_stride_size (int, optional): Stride size for the pooling. Defaults to 2.
        """
        super(Conv_Block_2d, self).__init__()

        self.conv_block_without_pool = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.pool_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride_size)
        )

    def forward(self, x):
        x_pre_pool = self.conv_block_without_pool(x)
        x_post_pool = self.pool_block(x_pre_pool)
        return x_pre_pool, x_post_pool


class Upconv_Block_2d(nn.Module):
    """Upconvolution Block for the expansive part of a UNet.

        Original ConvBlock would have the following parameters:
            conv_kernel_size = 3
            conv_padding_size = 0
            pool_kernel_size = 2
            pool_stride_size = 2
        But this would result in different input and output shapes.

    Args:
        -
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        conv_kernel_size=3,
        conv_padding_size=1,
        upconv_kernel_size=4,
        upconv_stride_size=2,
        upconv_padding_size=1,
    ):  # 3,0,2,2
        """
        Args:
            in_channels (int): Feature input channels for the first convolution.
            mid_channels (int): Feature input channels for the second convolution.
            out_channels (int): Feature output channels of the second convolution.
            conv_kernel_size (int, optional): Kernel size for the convolution. Defaults to 3.
            conv_padding_size (int, optional): Padding size for the convolution. Defaults to 1.
            upconv_kernel_size (int, optional): Kernel size for the upconvolution. Defaults to 4.
            upconv_stride_size (int, optional): Stride size for the upconvolution. Defaults to 2.
            upconv_padding_size (int, optional): Padding size for the upconvolution. Defaults to 1.
        """
        super(Upconv_Block_2d, self).__init__()

        self.upconv_block_with_transpose = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=upconv_kernel_size,
                stride=upconv_stride_size,
                padding=upconv_padding_size,
            ),
        )

    def forward(self, x):
        x = self.upconv_block_with_transpose(x)

        return x
