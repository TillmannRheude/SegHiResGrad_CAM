import torch
import torch.nn as nn

import torch.nn.functional as F

# 2D Block Imports
from blocks import Single_Conv_Block_2d, Conv_Block_2d, Upconv_Block_2d


#################################################################################################################################
############################################## 2D Network Architectures #########################################################
#################################################################################################################################
class Vanilla_UNet_2d(nn.Module):
    """Re-Implementation of the original proposed 2D UNet by Ronneberger et al. This implementation holds as a baseline for
        comparisons and evaluations.

        Citation:
        Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
        arXiv:1505.04597 [cs]. http://arxiv.org/abs/1505.04597

    Args:
        -
    """

    def __init__(
        self,
        encoder_channels=[1, 64, 128, 256, 512, 1024],
        decoder_channels=[1024, 512, 256, 128, 64, 1],
        grad_cam=False,
    ):
        super(Vanilla_UNet_2d, self).__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.grad_cam = grad_cam

        # Encoder
        self.encoder = nn.ModuleList(
            [
                Conv_Block_2d(encoder_channels[i], encoder_channels[i + 1])
                for i in range(len(encoder_channels) - 1)
            ]
        )

        # single upconvolution
        self.upconv = nn.ConvTranspose2d(
            in_channels=decoder_channels[0],
            out_channels=decoder_channels[1],
            kernel_size=2,
            stride=2,
        )

        # Decoder
        decoder_depth = len(decoder_channels) - 3
        self.decoder = nn.ModuleList(
            [
                Upconv_Block_2d(
                    in_channels=decoder_channels[i],
                    mid_channels=decoder_channels[i + 1],
                    out_channels=decoder_channels[i + 2],
                )
                for i in range(decoder_depth)
            ]
        )

        # Final layer
        self.final_layer = nn.ModuleList(
            [
                Single_Conv_Block_2d(
                    in_channels=decoder_channels[-3], out_channels=decoder_channels[-2]
                ),
                Single_Conv_Block_2d(
                    in_channels=decoder_channels[-2], out_channels=decoder_channels[-2]
                ),
                Single_Conv_Block_2d(
                    in_channels=decoder_channels[-2],
                    out_channels=decoder_channels[-1],
                    with_relu_batchnorm=False,
                    conv_kernel_size=1,
                    conv_padding_size=0,
                ),
            ]
        )

        self.sigmoid_layer = nn.Sigmoid()

        self.gradients = None  # placeholder for GradCam
        self.h = None  # placeholder for GradCam

    def pad_tensor(self, tensor_with_final_shapes, x):
        """Helper function.

        Args:
            tensor_with_final_shapes (torch.tensor): tensor with the shapes which x should have.
            x (torch.tensor): current tensor without the necessary (unpadded) shapes.

        Returns:
            torch.tensor: padded tensor x with shapes of tensor_with_final_shapes
        """
        # padding of the decoded tensor:
        _, _, h_e, w_e = tensor_with_final_shapes.shape
        _, _, h_x, w_x = x.shape

        if h_e > h_x:
            padded_tensor_x = F.pad(input=x, pad=(0, 0, 1, 0), mode="constant", value=0)
            if w_e > w_x:
                padded_tensor_x = F.pad(
                    input=padded_tensor_x, pad=(1, 0, 0, 0), mode="constant", value=0
                )
                return padded_tensor_x

        elif w_e > w_x:
            padded_tensor_x = F.pad(input=x, pad=(1, 0, 0, 0), mode="constant", value=0)

        else:
            padded_tensor_x = x.clone()

        return padded_tensor_x

    def activations_hook(self, grad):
        # for GradCam
        self.gradients = grad

    def forward(self, x, cam_level):
        # *----------------------------------------------- Encoder -----------------------------------------------*
        encoded_values = []
        for i, block in enumerate(self.encoder):
            # send the tensor to the encoder block and get the encoded tensor before and after the max pooling operation
            x_pre_pool, x_post_pool = block(x)
            if i == 0 and cam_level == 0 and self.grad_cam:
                self.h = x_pre_pool.register_hook(self.activations_hook)
            if i == 1 and cam_level == 1 and self.grad_cam:
                self.h = x_pre_pool.register_hook(self.activations_hook)
            if i == 2 and cam_level == 2 and self.grad_cam:
                self.h = x_pre_pool.register_hook(self.activations_hook)
            # for depth 5:
            #if i == 3 and cam_level == 3 and self.grad_cam:
            #    self.h = x_pre_pool.register_hook(self.activations_hook)

            # save the encoded tensor before the max pooling operation for the skip connection part later
            encoded_values.append(x_pre_pool)
            if i == 4:
                pass
            x = x_post_pool
        x = x_pre_pool
        if cam_level == 4 and self.grad_cam:
            self.h = x_pre_pool.register_hook(self.activations_hook)

        # delete the last skip connection tensor
        encoded_values = encoded_values[:-1]

        # for GradCam
        #if self.grad_cam and cam_level == 6:
        #    self.h = x_pre_pool.register_hook(self.activations_hook)

        # *----------------------------------------------- Decoder -----------------------------------------------*
        # first upconvolution in the decoder path
        x = self.upconv(x)

        for i, block in enumerate(self.decoder):
            # get the encoded value from the encoder path
            encoded_value = encoded_values.pop()
            # pad the decoding value to the necessary shape
            decoding_value_padded = self.pad_tensor(encoded_value, x)
            # concatenate the tensors
            x = torch.cat([decoding_value_padded, encoded_value], dim=1)
            # send the final concatenated tensor to the decoder block
            x = block(x)
            if i == 0 and cam_level == 5 and self.grad_cam:
                self.h = x.register_hook(self.activations_hook)
            if i == 1 and cam_level == 6 and self.grad_cam:
                self.h = x.register_hook(self.activations_hook)
            # for depth level 5: 
            #if i == 2 and cam_level == 7 and self.grad_cam:
            #    self.h = x.register_hook(self.activations_hook)


        # *-------------------------------------------- Final Layer ----------------------------------------------*
        # get the encoded value from the encoder path
        encoded_value = encoded_values.pop()
        # pad x to the necessary shape
        x_padded = self.pad_tensor(tensor_with_final_shapes=encoded_value, x=x)
        # concatenate the tensors
        x = torch.cat([x_padded, encoded_value], dim=1)

        # send the final concatenated tensor to the final layer block
        for i, block in enumerate(self.final_layer):
            x = block(x)
            if i == 1 and cam_level == 8 and self.grad_cam:
                self.h = x.register_hook(self.activations_hook)
        
        # *------------------------------------- Activation Function --------------------------------------------*
        if self.decoder_channels[-1] == 1:  # case of binary segmentation
            x = self.sigmoid_layer(x)

        return x

    def get_act_grad(self):
        # for GradCam
        return self.gradients

    def get_act(self, x, cam_level):
        """ 
        # for GradCam
        for i, block in enumerate(self.encoder):
            # send the tensor to the encoder block and get the encoded tensor before and after the max pooling operation
            x_pre_pool, x_post_pool = block(x)
            # save the encoded tensor before the max pooling operation for the skip connection part later
            if i == 4:
                pass
            x = x_post_pool
        x = x_pre_pool
        """

        # for GradCam
        encoded_values = []
        for i, block in enumerate(self.encoder):
            # send the tensor to the encoder block and get the encoded tensor before and after the max pooling operation
            x_pre_pool, x_post_pool = block(x)
            if i == 0 and cam_level == 0 and self.grad_cam:
                return x_pre_pool
            if i == 1 and cam_level == 1 and self.grad_cam:
                return x_pre_pool
            if i == 2 and cam_level == 2 and self.grad_cam:
                return x_pre_pool
            if i == 3 and cam_level == 3 and self.grad_cam:
                return x_pre_pool
            #if i == 4 and cam_level == 4 and self.grad_cam:
            #    return x_pre_pool
            # save the encoded tensor before the max pooling operation for the skip connection part later
            encoded_values.append(x_pre_pool)
            if i == 4:
                pass
            x = x_post_pool
        x = x_pre_pool

        if cam_level == 4 and self.grad_cam:
            return x_pre_pool

        # delete the last skip connection tensor
        encoded_values = encoded_values[:-1]

        # for GradCam
        #if self.grad_cam and cam_level == 6:
        #    self.h = x_pre_pool.register_hook(self.activations_hook)

        # *----------------------------------------------- Decoder -----------------------------------------------*
        # first upconvolution in the decoder path
        x = self.upconv(x)

        for i, block in enumerate(self.decoder):
            # get the encoded value from the encoder path
            encoded_value = encoded_values.pop()
            # pad the decoding value to the necessary shape
            decoding_value_padded = self.pad_tensor(encoded_value, x)
            # concatenate the tensors
            x = torch.cat([decoding_value_padded, encoded_value], dim=1)
            # send the final concatenated tensor to the decoder block
            x = block(x)
            if i == 0 and cam_level == 5 and self.grad_cam:
                return x
            if i == 1 and cam_level == 6 and self.grad_cam:
                return x
            if i == 2 and cam_level == 7 and self.grad_cam:
                return x

         # *-------------------------------------------- Final Layer ----------------------------------------------*
        # get the encoded value from the encoder path
        encoded_value = encoded_values.pop()
        # pad x to the necessary shape
        x_padded = self.pad_tensor(tensor_with_final_shapes=encoded_value, x=x)
        # concatenate the tensors
        x = torch.cat([x_padded, encoded_value], dim=1)

        # send the final concatenated tensor to the final layer block
        for i, block in enumerate(self.final_layer):
            x = block(x)
            if i == 1 and cam_level == 8 and self.grad_cam:
                return x

        # return x
