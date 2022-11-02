import torch.nn as nn
import torch.nn.functional as F
from .layers import Convx2, DownBlock, UpBlock, WithSE


class PlainUNet(nn.Module):
    """
    A straight-forward UNet implementation
    """
    tasks = ['seg']

    def __init__(self, config):
        super().__init__()
        bc = config['base_channels']
        batch_norm = config['batch_norm']
        padding_mode = config['padding_mode']
        conv_block = Convx2
        stack_height = config['stack_height']
        output_channels = config['output_channels']

        if config['squeeze_excitation']:
            conv_block = WithSE(conv_block)
        self.init = conv_block(config['input_channels'], bc, bn=batch_norm, padding_mode=padding_mode)

        conv_args = dict(
            conv_block=conv_block,
            bn=batch_norm,
            padding_mode=padding_mode
        )

        self.down_blocks = nn.ModuleList([
            DownBlock((1<<i)*bc, (2<<i)*bc, **conv_args)
            for i in range(stack_height)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock((2<<i)*bc, (1<<i)*bc, **conv_args)
            for i in reversed(range(stack_height))
        ])

        self.final = conv_block(bc, bc, bn=batch_norm, padding_mode=padding_mode)
        self.segment = nn.Conv2d(bc, output_channels, 1)

    def forward(self, x):
        x = self.init(x)

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)

        x = self.final(x)
        x = self.segment(x)

        return x
