"""Create a discriminator.

Parameters:
    input_nc (int)     -- the number of channels in input images
    ndf (int)          -- the number of filters in the first conv layer
    netD (str)         -- the architecture's name: basic | n_layers | pixel
    n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
    norm (str)         -- the type of normalization layers used in the network.
    init_type (str)    -- the name of the initialization method.
    init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

Returns a discriminator

Our current implementation provides three types of discriminators:
    [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
    It can classify whether 70×70 overlapping patches are real or fake.
    Such a patch-level discriminator architecture has fewer parameters
    than a full-image discriminator and can work on arbitrarily-sized images
    in a fully convolutional fashion.
    [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
    with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
    [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
    It encourages greater color diversity but has no effect on spatial statistics.
The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
"""

import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        self.input_channels = input_nc
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
