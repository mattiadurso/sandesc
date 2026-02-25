import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from pathlib import Path

# Add project root and external repos to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from model.modules import *


class SANDesc(nn.Module):
    def __init__(
        self,
        ch_in: int = 3,
        kernel_size: int = 5,
        activ: str = "gelu",
        norm: str = "batch",
        skip_connection: bool = False,
        spatial_attention: bool = False,
        third_block: bool = False,
        down_output_channels=[16, 32, 64, 64, 64],
        up_output_channels=[64, 64, 64, 128],
        **kwargs
    ):
        """
        Args:
            ch_in: int, number of input channels
            kernel_size: int, kernel size of the convolutional layers
            activ: str, activation function. Choose between 'relu', 'prelu', 'gelu'
            skip_connection: bool, if True, skip connections and a second unet block are added to the network
            spatial_attention: bool, if True, spatial attention is added to the network
            third_block: bool, if True, a third unet block is added to the network
            down_output_channels: list, number of channels of the output of each down block.
            up_output_channels: list, number of channels of the output of each up block. add +1 to get the same unet of disk in last element, eg [64, 64, 64, 128+1]
        Returns:
            des_vol: Tensor, descriptor volume. Shape [B, des_dim, H, W]

        The last element of up_output_channels is the number of channels of the descriptor.
        """

        super().__init__()
        self.conv_highest = nn.Conv2d(
            ch_in,
            down_output_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

        common = {
            "kernel_size": kernel_size,
            "activ": activ,
            "norm": norm,
            "skip_connection": skip_connection,  # and second block
            "spatial_attention": spatial_attention,
            "third_block": third_block,
        }

        self.down0 = Unet_down_block(
            down_output_channels[0], down_output_channels[1], **common
        )
        self.down1 = Unet_down_block(
            down_output_channels[1], down_output_channels[2], **common
        )
        self.down2 = Unet_down_block(
            down_output_channels[2], down_output_channels[3], **common
        )
        self.down3 = Unet_down_block(
            down_output_channels[3], down_output_channels[4], **common
        )

        self.up0 = Unet_up_block(
            down_output_channels[-1] + down_output_channels[-2],
            up_output_channels[0],
            **common
        )
        self.up1 = Unet_up_block(
            down_output_channels[-3] + up_output_channels[0],
            up_output_channels[1],
            **common
        )
        self.up2 = Unet_up_block(
            down_output_channels[-4] + up_output_channels[1],
            up_output_channels[2],
            **common
        )
        self.up3 = Unet_up_block(
            down_output_channels[-5] + up_output_channels[2],
            up_output_channels[3],
            kernel_size=kernel_size,
            activ=None,
            norm=None,
        )

    def load_weights(self, weights):
        """
        Load weights into the model.
        Args:
            weights (str): Path to the weights file.
        """
        weights = torch.load(weights, weights_only=False)
        self.load_state_dict((weights["state_dict"]))

    def forward(self, x: Tensor, _=None, normalize=True) -> Tuple[Tensor, Tensor]:

        x0 = self.conv_highest(x)  # B,c_in,H,W

        x1 = self.down0(x0)  # B,C1,H/2,W/2
        x2 = self.down1(x1)  # B,C2,H/4,W/4
        x3 = self.down2(x2)  # B,C3,H/8,W/8
        x4 = self.down3(x3)  # B,C4,H/16,W/16

        x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
        x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
        x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
        x8 = self.up3(x7, x0)  # B,des_dim,H,W

        if normalize:
            x8 = F.normalize(x8, p=2, dim=1)  # B,des_dim,H,W

        return x8


# main
if __name__ == "__main__":
    from torchinfo import summary

    device = "cuda"

    model = (
        SANDesc(skip_connection=True, spatial_attention=True, third_block=True)
        .to(device)
        .eval()
    )
    summary(model)

    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)

    print(out.shape)
