# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import torch.nn as nn


from block_config import ttypes as b_config
from nasrec.blocks import set_block_from_config
from .base_net import BaseNet
from .utils import (
    Optimizers,
    apply_emb,
    create_emb_dict,
    create_optimizers_for_dense,
    create_optimizers_for_embed,
)


logger = logging.getLogger(__name__)


class NASRecNet(BaseNet):
    def __init__(self, model_config, feature_config):
        super(NASRecNet, self).__init__(model_config, feature_config)

        self.nasrec_net_option = self.model_config.get_nasrec_net()
        self.num_block = len(self.nasrec_net_option.block_configs)
        self._init_model_params()
        self._build_arc()

    def _init_model_params(self):
        self.sparse_hash_size = {
            item.name: int(item.hash_size)
            for item in self.sparse_feature_options.features
        }
        self.feat_dim = {
            "dense": {0: [self.num_dense_feat]},
            "sparse": {
                0: [self.sparse_feature_options.embed_dim] * self.num_sparse_feat
            },
        }

    def _build_arc(self):
        self.emb_dict = create_emb_dict(self.sparse_feature_options)

        self.blocks = nn.ModuleList()
        for block_config in self.nasrec_net_option.block_configs:
            block = set_block_from_config(block_config, self.feat_dim)
            self.feat_dim = block.dim_config(self.feat_dim)
            self.blocks.append(block)

        # build up final block
        self.blocks.append(self._build_final_block())

    def _build_final_block(self):
        """Construct the final block
        """
        dense = deepcopy(self.feat_dim["dense"])
        sparse = deepcopy(self.feat_dim["sparse"])

        # make dicts of all features id (including intermidiate features)
        for block_id in dense:
            if len(dense[block_id]) > 0:
                dense[block_id] = list(range(dense[block_id][0]))
            else:
                dense[block_id] = []
        for block_id in sparse:
            sparse[block_id] = list(range(len(sparse[block_id])))

        # remove the features that has already been used as intermidiate input
        for block_id in range(0, self.num_block):
            dense_feat = self.blocks[block_id].feat_dense_id
            sparse_feat = self.blocks[block_id].feat_sparse_id
            for former_block_id in dense_feat:
                tmp_ids = dense_feat[former_block_id]
                dense[former_block_id] = (
                    (
                        []
                        if tmp_ids == [-1]
                        else list(set(dense[former_block_id]) - set(tmp_ids))
                    )
                    if former_block_id in dense
                    else []
                )
            for former_block_id in sparse_feat:
                tmp_ids = sparse_feat[former_block_id]
                sparse[former_block_id] = (
                    (
                        []
                        if tmp_ids == [-1]
                        else list(set(sparse[former_block_id]) - set(tmp_ids))
                    )
                    if former_block_id in sparse
                    else []
                )

        # convert feature dicts (dense & sparse) to feature configs
        feat_configs = []
        for block_id, feat_list in dense.items():
            if block_id in sparse:
                feat_config = b_config.FeatSelectionConfig(
                    block_id=block_id, dense=feat_list, sparse=sparse[block_id]
                )
            else:
                feat_config = b_config.FeatSelectionConfig(
                    block_id=block_id, dense=feat_list, sparse=[]
                )
            feat_configs.append(feat_config)
        for block_id, feat_list in sparse.items():
            if block_id in dense:
                continue
            else:
                feat_config = b_config.FeatSelectionConfig(
                    block_id=block_id, dense=[], sparse=feat_list
                )
            feat_configs.append(feat_config)

        # construct the MLP block config
        block_config = b_config.BlockConfig(
            mlp_block=b_config.MLPBlockConfig(
                name="MLPBlock",
                block_id=self.num_block + 1,
                arc=[1],
                type=b_config.BlockType(dense=b_config.DenseBlockType()),
                input_feat_config=feat_configs,
                ly_act=False,
            )
        )
        return set_block_from_config(block_config, self.feat_dim)

    def get_optimizers(self):
        optimizers = Optimizers()
        # add dense optimizers
        create_optimizers_for_dense(
            optimizers,
            named_parameters=self.named_parameters(),
            dense_optim_config=self.dense_feature_options.optim,
        )
        # add sparse optimizers
        create_optimizers_for_embed(
            optimizers,
            emb_dict=self.emb_dict,
            sparse_feature_options=self.sparse_feature_options,
        )
        return optimizers

    def forward(self, feats):
        # process sparse features(using embeddings), resulting in a list of row vectors
        feat_dict = {"dense": {0: feats["dense"]}}  #  if self.num_dense_feat > 0 else []
        ly = apply_emb(feats, self.emb_dict, self.sparse_hash_size)
        feat_dict["sparse"] = {
            0: {feat_id: ly[feat_id] for feat_id in range(self.num_sparse_feat)}
        }
        # blocks
        for qq, block in enumerate(self.blocks):
            feat_dict = block(feat_dict)
        p = feat_dict["dense"][self.blocks[-1].block_id]
        return p.view(-1)
