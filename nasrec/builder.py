# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch

from config import ttypes as config
from .evolutionary_controller import EvolutionaryController
from .random_controller import RandomController


logger = logging.getLogger(__name__)


def build_searcher(searcher_config, feature_config):
    if searcher_config.getType() == config.SearcherConfig.RANDOM_SEARCHER:
        return build_random_searcher(searcher_config, feature_config)
    elif searcher_config.getType() == config.SearcherConfig.EVOLUTIONARY_SEARCHER:
        return build_evolutionary_searcher(searcher_config, feature_config)
    else:
        raise ValueError("Unknown searcher type.")


def build_random_searcher(searcher_config, feature_config):
    return RandomController(
        searcher_config=searcher_config, feature_config=feature_config
    )


def build_evolutionary_searcher(searcher_config, feature_config):
    return EvolutionaryController(
        searcher_config=searcher_config, feature_config=feature_config
    )


def save_searcher(filename, searcher):
    logger.info("Saving searcher to {}".format(filename))
    state = {
        "state_dict": searcher.state_dict(),
        "searcher_config": searcher.searcher_config,
        "feature_config": searcher.feature_config,
    }
    torch.save(state, filename)


def load_searcher(filename):
    logger.info("Loading searcher from {}".format(filename))
    state = torch.load(filename)
    searcher_config = state["searcher_config"]
    feature_config = state["feature_config"]
    searcher = build_searcher(
        searcher_config=searcher_config, feature_config=feature_config
    )
    searcher.load_state_dict(state["state_dict"])
    return searcher
