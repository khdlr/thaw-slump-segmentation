from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .manet import MAnet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN
from .plain_unet import PlainUNet
import torch.nn as nn

from . import encoders
from . import utils
from . import losses

from .__version__ import __version__

from typing import Optional
import torch


def create_loss(
    name: str,
) -> torch.nn.Module:
    """LossFn wrapper. Allows to create any loss_fn just with parametes"""

    losses_list = [
        losses.JaccardLoss,
        losses.DiceLoss,
        losses.FocalLoss,
        losses.LovaszLoss,
        losses.SoftBCEWithLogitsLoss,
        losses.SoftCrossEntropyLoss
    ]
    losses_dict = {l.__name__.lower(): l for l in losses_list}

    builtin_losses = [nn.BCELoss]
    builtin_losses_dict = {l.__name__.lower(): l for l in builtin_losses}
    name = name.lower()
    if name in losses_dict:
        return losses_dict[name](mode=losses.BINARY_MODE, ignore_index=255)
    elif name in builtin_losses_dict:
        return builtin_losses_dict[name]()
    raise KeyError("Wrong loss type `{}`. Available options are: {}".format(
        name, list(builtin_losses_dict.keys()) + list(losses_dict.keys()),
    ))
