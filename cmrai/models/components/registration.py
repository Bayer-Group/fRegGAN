# system
import os

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

# local
from cmrai.models.components import layers

sampling_align_corners = False

# The number of filters in each block of the encoding part (down-sampling).
ndf = {
    "A": [32, 64, 64, 64, 64, 64, 64],
}
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {
    "A": [64, 64, 64, 64, 64, 64, 32],
}
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {
    "A": True,
}
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {
    "A": 3,
}
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {
    "A": True,
}
# The activation used in the down-sampling path.
down_activation = {
    "A": "leaky_relu",
}
# The activation used in the up-sampling path.
up_activation = {
    "A": "leaky_relu",
}


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super().__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(
                self,
                f"down_{conv_num}",
                layers.DownBlock(
                    in_nf,
                    out_nf,
                    3,
                    1,
                    1,
                    activation=act,
                    init_func=init_func,
                    bias=True,
                    use_resnet=use_down_resblocks[cfg],
                    use_norm=False,
                ),
            )
            skip_nf[f"down_{conv_num}"] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = layers.Conv(
                in_nf,
                2 * in_nf,
                1,
                1,
                0,
                activation=act,
                init_func=init_func,
                bias=True,
                use_resnet=False,
                use_norm=False,
            )
            self.t = (
                (lambda x: x)
                if resnet_nblocks[cfg] == 0
                else layers.ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func)
            )
            self.c2 = layers.Conv(
                2 * in_nf,
                in_nf,
                1,
                1,
                0,
                activation=act,
                init_func=init_func,
                bias=True,
                use_resnet=False,
                use_norm=False,
            )
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(
                self,
                f"up_{conv_num}",
                layers.Conv(
                    in_nf + skip_nf[f"down_{conv_num}"],
                    out_nf,
                    3,
                    1,
                    1,
                    bias=True,
                    activation=act,
                    init_fun=init_func,
                    use_norm=False,
                    use_resnet=False,
                ),
            )
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(
                layers.ResnetTransformer(in_nf, 1, init_func),
                layers.Conv(
                    in_nf,
                    in_nf,
                    1,
                    1,
                    0,
                    use_resnet=False,
                    init_func=init_func,
                    activation=act,
                    use_norm=False,
                ),
            )
        else:
            self.refine = lambda x: x
        self.output = layers.Conv(
            in_nf,
            2,
            3,
            1,
            1,
            use_resnet=False,
            bias=True,
            init_func=("zeros" if init_to_identity else init_func),
            activation=None,
            use_norm=False,
        )

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, f"down_{conv_num}")(x)
            skip_vals[f"down_{conv_num}"] = skip
            conv_num += 1
        if hasattr(self, "t"):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals[f"down_{conv_num}"]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode="bilinear")
            x = torch.cat([x, s], 1)
            x = getattr(self, f"up_{conv_num}")(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class SpatialDeformation(nn.Module):
    def __init__(self, batch_size, image_size):
        super().__init__()
        size = (image_size, image_size)
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        self.grid = torch.unsqueeze(grid, 0)

    # @staticmethod
    def forward(self, src, flow):
        grid = self.grid.type_as(flow)
        # new locations
        new_locs = grid + flow
        shape = flow.shape[2:]
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")
        return warped


class DeformationFieldNet(nn.Module):
    def __init__(self, image_size, input_channels):
        super().__init__()
        # height,width=256,256
        # in_channels_a,in_channels_b=1,1
        init_func = "kaiming"
        init_to_identity = True

        # paras end------------
        if isinstance(image_size, tuple):
            self.oh, self.ow = image_size
        else:
            self.oh = image_size
            self.ow = image_size
        self.input_channels = input_channels
        self.offset_map = ResUnet(
            self.input_channels,
            self.input_channels,
            cfg="A",
            init_func=init_func,
            init_to_identity=init_to_identity,
        )
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):
        deformations = self.offset_map(img_a, img_b)
        return deformations
