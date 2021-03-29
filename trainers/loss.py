# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch

from config import ttypes as config


logger = logging.getLogger(__name__)


def build_loss(model, loss_config):
    if loss_config.getType() == config.LossConfig.BCEWITHLOGITS:
        logger.warning(
            "Creating BCEWithLogitsLoss: {}".format(loss_config.get_bcewithlogits())
        )
        return torch.nn.BCEWithLogitsLoss(reduction="none")
    elif loss_config.getType() == config.LossConfig.MSE:
        logger.warning("Creating MSELoss: {}".format(loss_config.get_mse()))
        return torch.nn.MSELoss(reduction="none")
    elif loss_config.getType() == config.LossConfig.BCE:
        logger.warning("Creating BCELoss: {}".format(loss_config.get_bce()))
        return torch.nn.BCELoss(reduction="none")
    else:
        raise ValueError("Unknown loss type.")


# TODO add equal weight training and calibration for ads data
def apply_loss(loss, pred, label, weight=None):
    E = loss(pred, label)
    return torch.mean(E) if weight is None else torch.mean(E * weight.view(-1))
