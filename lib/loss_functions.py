# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.nn.functional as F


def get_loss(loss_args):
    loss_type = loss_args['type']
    if loss_type in ('CrossEntropy') :
        loss_class = torch.nn.BCEWithLogitsLoss
        args = dict()
        if 'weights' in loss_args:
            args['weight'] = torch.tensor(loss_args['weights'])
    elif loss_type == 'AutoCE':
        return auto_ce
    else:
        print(f"No Loss of type {loss_type} known")

    return loss_class(**args)


def auto_ce(y_hat, y):
    with torch.no_grad():
        C = y_hat.shape[1]
        counts = torch.stack([(y == i).float().mean() for i in range(C)])
        weights = 1.0 / (C * counts)
    return F.cross_entropy(y_hat, y.squeeze(1), weight=weights, ignore_index=255)
