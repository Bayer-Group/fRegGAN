import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(
        self,
        input_channels,
        n_residual_blocks,
        use_conv_transpose=True,
        use_output_activation=True,
        input_size=(256, 256),
    ):
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.input_channels = int(input_channels)
        # Initial convolution block
        model_head = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            if use_conv_transpose:
                model_tail += [
                    nn.ConvTranspose2d(
                        in_features,
                        out_features,
                        3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
            else:
                model_tail += [
                    nn.Upsample(scale_factor=2),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        in_features,
                        out_features,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_channels, 7),
        ]
        if use_output_activation:
            model_tail += [nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x
