# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('gen-py')

import logging

import torch

from config import ttypes as config
from .nas_modules import NASRecNet


logger = logging.getLogger(__name__)


def build_model(model_config, feature_config):
    if model_config.getType() == config.ModelConfig.NASREC_NET:
        return build_nasrec_net(model_config, feature_config)
    else:
        raise ValueError("Unknown model type.")


def build_nasrec_net(model_config, feature_config):
    return NASRecNet(model_config=model_config, feature_config=feature_config)


def save_model(filename, model):
    logger.warning("Saving model to {}".format(filename))
    state = {
        "state_dict": model.state_dict(),
        "model_config": model.model_config,
        "feature_config": model.feature_config,
    }
    torch.save(state, filename)


def load_model(filename):
    logger.warning("Loading model from {}".format(filename))
    state = torch.load(filename, map_location='cpu')
    model_config = state["model_config"]
    feature_config = state["feature_config"]
    model = build_model(model_config=model_config, feature_config=feature_config)
    model.load_state_dict(state["state_dict"])
    return model
