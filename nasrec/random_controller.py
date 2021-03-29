# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from config import ttypes as config
from models.nas_modules import NASRecNet
from .base_searcher import BaseSearcher


logger = logging.getLogger(__name__)


class RandomController(BaseSearcher):
    def __init__(self, searcher_config, feature_config):
        super(RandomController, self).__init__(searcher_config, feature_config)
        self.controller_option = searcher_config.get_random_searcher()
        self._init_base_searcher_params()

    def _build_arc(self):
        pass

    def sample(self, batch_size=1, return_config=False):
        """Samples a batch_size number of NasRecNets from the controller, where
        each node is made up of a set of blocks with number self.num_blocks
        """
        if batch_size < 1:
            raise ValueError("Wrong batch_size.")

        nasrec_nets, all_vec_configs, nasrec_arc_vecs = [], [], []
        for _ in range(batch_size):
            vecs, vec_configs = self.random_sample()
            arc_vec = np.concatenate(vecs)
            nasrec_arc_vecs.append(arc_vec)
            all_vec_configs.append(vec_configs)
            block_configs = self.vecs_to_model_config(vec_configs)
            model_config = config.ModelConfig(
                nasrec_net=config.NASRecNetConfig(block_configs=block_configs)
            )
            if return_config:
                nasrec_nets.append(model_config)
            else:
                nasrec_nets.append(NASRecNet(model_config, self.feature_config))

        return nasrec_nets, [], all_vec_configs, nasrec_arc_vecs

    def update(self, probs, rewards):
        pass
