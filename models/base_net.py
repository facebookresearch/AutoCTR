# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import torch.nn as nn


logger = logging.getLogger(__name__)


class BaseNet(nn.Module):
    def __init__(self, model_config, feature_config):
        super(BaseNet, self).__init__()

        # for serilization purpose
        self.model_config = deepcopy(model_config)
        self.feature_config = deepcopy(feature_config)

        self.dense_feature_options = self.feature_config.dense
        self.sparse_feature_options = self.feature_config.sparse

        self.num_dense_feat = len(self.dense_feature_options.features)
        self.num_sparse_feat = len(self.sparse_feature_options.features)

    def _build_arc(self):
        raise NotImplementedError

    def get_optimizers(self):
        raise NotImplementedError

    def forward(self, fs):
        raise NotImplementedError
