# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from block_config import ttypes as b_config
from models.utils import create_mlp
from .utils import (
    cat_feats,
    clean_feat_id,
    config_to_dict,
    convert_to_emb,
    create_cin,
    create_crossnet,
    create_emb_converter,
    create_transformer,
    extract_dense_feat,
    get_sparse_feat_dim_num,
)


logger = logging.getLogger(__name__)


def set_block_from_config(block_config, feat_dim):
    if block_config is None:
        return None

    name2block = {
        b_config.BlockConfig.MLP_BLOCK: MLPBlock,
        b_config.BlockConfig.CROSSNET_BLOCK: CrossNetBlock,
        b_config.BlockConfig.FM_BLOCK: FMBlock,
        b_config.BlockConfig.DOTPROCESSOR_BLOCK: DotProcessorBlock,
        b_config.BlockConfig.CAT_BLOCK: CatBlock,
        b_config.BlockConfig.CIN_BLOCK: CINBlock,
        b_config.BlockConfig.ATTENTION_BLOCK: AttentionBlock,
    }

    block_name = block_config.getType()  # block_config.name
    block = name2block[block_name]
    return block(block_config, feat_dim)


def save_block(block, filename):
    logger.info("Saving block to {}".format(filename))
    state = {
        "state_dict": block.state_dict(),
        "block_config": block.block_config,
        "feat_dim": {"dense": block.feat_dense_dim, "sparse": block.feat_sparse_dim},
    }
    torch.save(state, filename)


def load_block(filename):
    logger.info("Loading model from {}".format(filename))
    state = torch.load(filename)
    block_config = state["block_config"]
    feat_dim = state["feat_dim"]
    block = set_block_from_config(block_config=block_config, feat_dim=feat_dim)
    block.load_state_dict(state["state_dict"])
    return block


class BaseBlock(nn.Module):
    def __init__(self, block_config, feat_dim):
        super(BaseBlock, self).__init__()

        # for serilization purpose
        self.block_config = deepcopy(block_config)
        # extract input feat_dim dictionary {block_id: feat_dim (list)}
        self.feat_dense_dim = feat_dim["dense"]
        self.feat_sparse_dim = feat_dim["sparse"]

    def _init_basic_block_params(self):
        self.block_id = self.block_option.block_id

        self.input_feat_config = self.block_option.input_feat_config

        # convert input feat_id into dictionary format {block_id: feat_id (list)}
        self.feat_dense_id, self.feat_sparse_id = config_to_dict(
            self.block_option.input_feat_config
        )

        # check and modify feat_ids
        self.feat_dense_id = clean_feat_id(
            self.feat_dense_id, self.feat_dense_dim, "dense"
        )
        self.feat_sparse_id = clean_feat_id(
            self.feat_sparse_id, self.feat_sparse_dim, "sparse"
        )

        # get input feature number
        # dense feature
        self.num_dense_feat = sum(
            (
                self.feat_dense_dim[b][0]  # all dense feats in block b
                if self.feat_dense_id[b] == [-1]
                else len(self.feat_dense_id[b])
            )
            for b in self.feat_dense_id
        )
        self.num_sparse_feat = sum(
            (
                len(self.feat_sparse_dim[b])
                if self.feat_sparse_id[b] == [-1]
                else len(self.feat_sparse_id[b])
            )
            for b in self.feat_sparse_id
        )

    def _refine_emb_arc(self):
        # refine the arc if the raw input dense feature are treated as sparse
        # treat input dense features in block 0 as sparse features if existed
        self.dense_as_sparse_id, self.num_dense_as_sparse_feat = None, 0
        if self.emb_config.dense_as_sparse and 0 in self.feat_dense_id:
            self.dense_as_sparse_id = self.feat_dense_id.pop(0)
            self.num_dense_as_sparse_feat = (
                self.feat_dense_dim[0][0]
                if self.dense_as_sparse_id == [-1]
                else len(self.dense_as_sparse_id)
            )
            self.num_dense_feat -= self.num_dense_as_sparse_feat
            self.num_sparse_feat += self.num_dense_as_sparse_feat

    def forward(self, feat_dict):
        raise NotImplementedError

    def dim_config(self, feat_dim):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__[:-5]


class MLPBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(MLPBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_mlp_block()
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            # set mlp layer
            self.num_input_feat = self.num_dense_feat
            if self.num_sparse_feat > 0:
                self.num_input_feat += get_sparse_feat_dim_num(
                    self.feat_sparse_id, self.feat_sparse_dim
                )
            self.layers = create_mlp(
                [self.num_input_feat] + self.block_option.arc,
                ly_act=self.block_option.ly_act,
            )
        elif self.block_option.type.getType() == b_config.BlockType.EMB:

            self.emb_config = self.block_option.type.get_emb()

            self._refine_emb_arc()

            # set embeding layer
            self.feat_emb = create_emb_converter(
                self.num_dense_feat,
                self.feat_sparse_id,
                self.feat_sparse_dim,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )

            # set mlp layer
            self.layers = create_mlp(
                [self.emb_config.comm_embed_dim] + self.block_option.arc,
                ly_act=self.block_option.ly_act,
            )
        else:
            raise ValueError("Unsupported configuration for MLPBlock type.")

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            if self.block_option.type.getType() == b_config.BlockType.DENSE:
                feat_dim["dense"][self.block_id] = [self.block_option.arc[-1]]
            elif self.block_option.type.getType() == b_config.BlockType.EMB:
                if self.num_dense_feat > 0:
                    feat_dim["sparse"][self.block_id] = [self.block_option.arc[-1]] * (
                        self.num_sparse_feat + 1
                    )
                else:
                    feat_dim["sparse"][self.block_id] = [
                        self.block_option.arc[-1]
                    ] * self.num_sparse_feat
        return feat_dim

    def forward(self, feat_dict):

        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            feat = cat_feats(extracted_feat_dict, self.feat_sparse_id)
            try:
                p = self.layers(feat)
            except:
                exit()
            feat_dict["dense"][self.block_id] = p
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            if self.num_dense_as_sparse_feat > 0:
                extracted_feat_dict["dense_as_sparse"] = (
                    feat_dict["dense"][0]
                    if self.dense_as_sparse_id == [-1]
                    else feat_dict["dense"][0][:, self.dense_as_sparse_id]
                )
            feat = convert_to_emb(
                extracted_feat_dict,
                self.feat_emb,
                self.num_dense_feat,
                self.feat_sparse_id,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
            p = self.layers(feat)
            feat_dict["sparse"][self.block_id] = {
                feat_id: p[:, feat_id] for feat_id in range(p.shape[1])  # 1 for dense
            }
        return feat_dict

    def __str__(self):
        return (
            super().__str__()
            + "("
            + ", ".join(str(item) for item in self.block_option.arc)
            + ")"
        )


class CrossNetBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(CrossNetBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_crossnet_block()
        self.num_of_layers = self.block_option.num_of_layers
        self._init_basic_block_params()
        self._init_cross_params()
        self._build_arc()

    def _init_cross_params(self):
        # cross input feat id
        self.cross_feat_config = self.block_option.cross_feat_config

        # convert cross input feat_id into dictionary format {block_id: feat_id (list)}
        self.cross_feat_dense_id, self.cross_feat_sparse_id = config_to_dict(
            self.block_option.cross_feat_config
        )

        # check and modify feat_ids
        self.cross_feat_dense_id = clean_feat_id(
            self.cross_feat_dense_id, self.feat_dense_dim, "dense"
        )
        self.cross_feat_sparse_id = clean_feat_id(
            self.cross_feat_sparse_id, self.feat_sparse_dim, "dense"
        )

        # get cross input feature number
        # dense feature
        self.cross_num_dense_feat_per_block = []
        for b in self.cross_feat_dense_id:
            self.cross_num_dense_feat_per_block += (
                self.feat_dense_dim[b]  # all dense feats in block b
                if self.cross_feat_dense_id[b] == [-1]
                else [len(self.cross_feat_dense_id[b])]
            )
        # sparse feature
        self.cross_num_sparse_feat_per_block = []
        for b in self.cross_feat_sparse_id:
            self.cross_num_sparse_feat_per_block += (
                self.feat_sparse_dim[b]
                if self.cross_feat_sparse_id[b] == [-1]
                else [len(self.cross_feat_sparse_id[b])]
            )
        self.cross_num_dense_feat = sum(self.cross_num_dense_feat_per_block)
        self.cross_num_sparse_feat = sum(self.cross_num_sparse_feat_per_block)

        # remodify feat_ids if the block is emtpy block
        if (
            self.num_sparse_feat + self.num_dense_feat == 0
            or self.cross_num_dense_feat + self.cross_num_sparse_feat == 0
        ):
            self.feat_dense_id = {}
            self.feat_sparse_id = {}
            self.cross_feat_dense_id = {}
            self.cross_feat_sparse_id = {}

    def _build_arc(self):
        if (
            self.num_sparse_feat + self.num_dense_feat == 0
            or self.cross_num_dense_feat + self.cross_num_sparse_feat == 0
        ):
            return
        self.num_input_feat = self.num_dense_feat
        self.num_input_feat += get_sparse_feat_dim_num(
            self.feat_sparse_id, self.feat_sparse_dim
        )
        self.cross_num_input_feat = self.cross_num_dense_feat
        self.cross_num_input_feat += get_sparse_feat_dim_num(
            self.cross_feat_sparse_id, self.feat_sparse_dim
        )
        if self.num_input_feat != self.cross_num_input_feat:
            # construct a embedding layer
            self.emb_layer = nn.Linear(self.cross_num_input_feat, self.num_input_feat)
        self.weight_w, self.weight_b, self.batchnorm = create_crossnet(
            self.num_of_layers, self.num_input_feat
        )

    def dim_config(self, feat_dim):
        if (
            self.num_sparse_feat + self.num_dense_feat != 0
            and self.cross_num_dense_feat + self.cross_num_sparse_feat != 0
        ):
            feat_dim["dense"][self.block_id] = [self.num_input_feat]
        return feat_dim

    def forward(self, feat_dict):
        if (
            self.num_sparse_feat + self.num_dense_feat == 0
            or self.cross_num_dense_feat + self.cross_num_sparse_feat == 0
        ):
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        cross_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.cross_feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        # concatenate two feature dicts into two vectors
        feat = cat_feats(extracted_feat_dict, self.feat_sparse_id)
        cross_feat = cat_feats(cross_feat_dict, self.cross_feat_sparse_id)

        # crossnet
        if self.num_input_feat != self.cross_num_input_feat:
            cross_feat = self.emb_layer(cross_feat)
        for i in range(self.num_of_layers):
            feat = cross_feat * self.weight_w[i](feat) + self.weight_b[i] + feat
            if self.block_option.batchnorm:
                feat = self.batchnorm[i](feat)

        feat_dict["dense"][self.block_id] = feat
        return feat_dict

    def __str__(self):
        return super().__str__()


class FMBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(FMBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_fm_block()
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            self.num_input_feat = self.num_dense_feat
            if self.num_sparse_feat > 0:
                self.num_input_feat += get_sparse_feat_dim_num(
                    self.feat_sparse_id, self.feat_sparse_dim
                )
            # set FM layer
            # first order embedding layer
            self.weight_w_first = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )
            self.weight_b_first = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )
            # second order embedding layer
            self.weight_w_second = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )
            self.weight_b_second = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )

        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            self.emb_config = self.block_option.type.get_emb()

            self._refine_emb_arc()

            # set FM layer
            # first order embedding layer
            self.first_order_feat_emb = create_emb_converter(
                self.num_dense_feat,
                self.feat_sparse_id,
                self.feat_sparse_dim,
                1,
                self.num_dense_as_sparse_feat,
            )
            # second order embedding layer
            self.second_order_feat_emb = create_emb_converter(
                self.num_dense_feat,
                self.feat_sparse_id,
                self.feat_sparse_dim,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
        else:
            raise ValueError("Unsupported configuration for FMBlock type.")

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            feat_dim["dense"][self.block_id] = [1]
        return feat_dim

    def forward(self, feat_dict):

        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        # compute FM layer
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            feat = cat_feats(extracted_feat_dict, self.feat_sparse_id)
            feat1 = feat * self.weight_w_first + self.weight_b_first
            feat2 = feat * self.weight_w_second + self.weight_b_second
            p = self.fm_sum(feat1, feat2)
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            if self.num_dense_as_sparse_feat > 0:
                extracted_feat_dict["dense_as_sparse"] = (
                    feat_dict["dense"][0]
                    if self.dense_as_sparse_id == [-1]
                    else feat_dict["dense"][0][:, self.dense_as_sparse_id]
                )
            feat1 = convert_to_emb(
                extracted_feat_dict,
                self.first_order_feat_emb,
                self.num_dense_feat,
                self.feat_sparse_id,
                1,
                self.num_dense_as_sparse_feat,
            )
            feat2 = convert_to_emb(
                extracted_feat_dict,
                self.second_order_feat_emb,
                self.num_dense_feat,
                self.feat_sparse_id,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
            p = self.fm_sum(feat1, feat2)
        feat_dict["dense"][self.block_id] = p
        return feat_dict

    def fm_sum(self, feat1, feat2):
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            # first order
            p1 = torch.sum(feat1, 1)
            # second order
            sum_square = torch.pow(torch.sum(feat2, 1), 2)
            square_sum = torch.sum(torch.pow(feat2, 2), 1)
            p2 = (sum_square - square_sum) * 0.5
            p = p1 + p2
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            p1 = torch.sum(feat1, [1, 2])
            sum_square = torch.pow(torch.sum(feat2, 1), 2)
            square_sum = torch.sum(torch.pow(feat2, 2), 1)
            p2 = (sum_square - square_sum) * 0.5
            p = p1 + torch.sum(p2, 1)
        return p[:, None]

    def __str__(self):
        return super().__str__()


class DotProcessorBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(DotProcessorBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_dotprocessor_block()
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            self.num_input_feat = self.num_dense_feat
            if self.num_sparse_feat > 0:
                self.num_input_feat += get_sparse_feat_dim_num(
                    self.feat_sparse_id, self.feat_sparse_dim
                )
            # set DP layer
            self.weight_w = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )
            self.weight_b = nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.num_input_feat))
            )

        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            self.emb_config = self.block_option.type.get_emb()

            self._refine_emb_arc()

            self.num_input_feat = 1 + self.num_sparse_feat
            # set Embedding Layer
            self.feat_emb = create_emb_converter(
                self.num_dense_feat,
                self.feat_sparse_id,
                self.feat_sparse_dim,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
        else:
            raise ValueError("Unsupported configuration for DotProcessorBlock type.")

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            feat_dim["dense"][self.block_id] = [
                int(self.num_input_feat * (self.num_input_feat + 1) / 2)
            ]
        return feat_dim

    def forward(self, feat_dict):

        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        # compute DP layer
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            feat = cat_feats(extracted_feat_dict, self.feat_sparse_id)
            feat = feat * self.weight_w + self.weight_b
            p = self.dp_sum(feat[:, :, None])
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            if self.num_dense_as_sparse_feat > 0:
                extracted_feat_dict["dense_as_sparse"] = (
                    feat_dict["dense"][0]
                    if self.dense_as_sparse_id == [-1]
                    else feat_dict["dense"][0][:, self.dense_as_sparse_id]
                )
            feat = convert_to_emb(
                extracted_feat_dict,
                self.feat_emb,
                self.num_dense_feat,
                self.feat_sparse_id,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
            p = self.dp_sum(feat)
        feat_dict["dense"][self.block_id] = p
        return feat_dict

    def dp_sum(self, feat):
        Z = torch.matmul(feat, torch.transpose(feat, 1, 2))
        Zflat = Z.view((feat.shape[0], -1))
        num_ints = int(self.num_input_feat * (self.num_input_feat + 1) / 2)
        return Zflat[:, :num_ints]

    def __str__(self):
        return super().__str__()


class CatBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(CatBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_cat_block()
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            self.num_input_feat = self.num_dense_feat
            if self.num_sparse_feat > 0:
                self.num_input_feat += get_sparse_feat_dim_num(
                    self.feat_sparse_id, self.feat_sparse_dim
                )
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            self.emb_config = self.block_option.type.get_emb()

            self._refine_emb_arc()

            self.num_input_feat = 1 + self.num_sparse_feat
            # set Embedding Layer
            self.feat_emb = create_emb_converter(
                self.num_dense_feat,
                self.feat_sparse_id,
                self.feat_sparse_dim,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
        else:
            raise ValueError("Unsupported configuration for CatBlock type.")

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            if self.block_option.type.getType() == b_config.BlockType.DENSE:
                feat_dim["dense"][self.block_id] = [self.num_input_feat]
            elif self.block_option.type.getType() == b_config.BlockType.EMB:
                feat_dim["sparse"][self.block_id] = (
                    [self.emb_config.comm_embed_dim] * (self.num_sparse_feat + 1)
                    if self.num_dense_feat > 0
                    else [self.emb_config.comm_embed_dim] * self.num_sparse_feat
                )
        return feat_dim

    def forward(self, feat_dict):

        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        # compute Cat layer
        if self.block_option.type.getType() == b_config.BlockType.DENSE:
            p = cat_feats(extracted_feat_dict, self.feat_sparse_id)
            feat_dict["dense"][self.block_id] = (
                p[:, None] if self.num_input_feat == 1 else p
            )
        elif self.block_option.type.getType() == b_config.BlockType.EMB:
            if self.num_dense_as_sparse_feat > 0:
                extracted_feat_dict["dense_as_sparse"] = (
                    feat_dict["dense"][0]
                    if self.dense_as_sparse_id == [-1]
                    else feat_dict["dense"][0][:, self.dense_as_sparse_id]
                )
            p = convert_to_emb(
                extracted_feat_dict,
                self.feat_emb,
                self.num_dense_feat,
                self.feat_sparse_id,
                self.emb_config.comm_embed_dim,
                self.num_dense_as_sparse_feat,
            )
            feat_dict["sparse"][self.block_id] = {
                feat_id: p[:, feat_id] for feat_id in range(p.shape[1])  # 1 for dense
            }
        return feat_dict

    def __str__(self):
        return super().__str__()


class CINBlock(BaseBlock):
    """Compressed Interaction Network used in xDeepFM.
        https://arxiv.org/pdf/1803.05170.pdf.
    """

    def __init__(self, block_config, feat_dim):
        super(CINBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_cin_block()
        self.layer_sizes = self.block_option.arc
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return

        self.emb_config = self.block_option.emb_config

        self._refine_emb_arc()

        self.field_nums = [self.num_sparse_feat + 1]
        for i, size in enumerate(self.layer_sizes):
            if self.block_option.split_half:
                if i != len(self.layer_sizes) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True"
                    )

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        # set embeding layers
        self.feat_emb = create_emb_converter(
            self.num_dense_feat,
            self.feat_sparse_id,
            self.feat_sparse_dim,
            self.emb_config.comm_embed_dim,
            self.num_dense_as_sparse_feat,
        )

        # set CIN convolutional layers
        self.conv_layers, self.bias_layers, self.activation_layers = create_cin(
            self.layer_sizes, self.field_nums
        )

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            feat_dim["dense"][self.block_id] = (
                [sum(self.layer_sizes[:-1]) // 2 + self.layer_sizes[-1]]
                if self.block_option.split_half
                else [sum(self.layer_sizes)]
            )
        return feat_dim

    def forward(self, feat_dict):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        if self.num_dense_as_sparse_feat > 0:
            extracted_feat_dict["dense_as_sparse"] = (
                feat_dict["dense"][0]
                if self.dense_as_sparse_id == [-1]
                else feat_dict["dense"][0][:, self.dense_as_sparse_id]
            )

        # get feature matrix X0
        feat = convert_to_emb(
            extracted_feat_dict,
            self.feat_emb,
            self.num_dense_feat,
            self.feat_sparse_id,
            self.emb_config.comm_embed_dim,
            self.num_dense_as_sparse_feat,
        )
        if feat.dim() != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (feat.dim())
            )
        p = self.cin(feat)
        feat_dict["dense"][self.block_id] = p
        return feat_dict

    def cin(self, feat):
        dim = feat.shape[-1]
        p = []
        hidden_nn_layers = [feat]
        cross_feats = torch.split(hidden_nn_layers[0], dim * [1], 2)
        for l_idx, layer_size in enumerate(self.layer_sizes):
            curr_feats = torch.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = torch.stack(
                [
                    torch.bmm(curr_feats[t_idx], t.transpose(1, 2))
                    for t_idx, t in enumerate(cross_feats)
                ]
            )

            dot_result_m = dot_result_m.view(
                -1, 1, dot_result_m.shape[2], dot_result_m.shape[3]
            )
            # apply conv, add bias, activation
            curr_out = torch.squeeze(self.conv_layers[l_idx](dot_result_m))
            curr_out = curr_out.view(dim, -1, layer_size)  # (dim * batch_size * Hk)

            curr_out = curr_out + self.bias_layers[l_idx]
            curr_out = self.activation_layers[l_idx](curr_out)
            curr_out = curr_out.permute(1, 2, 0)

            if self.block_option.split_half:
                if l_idx != len(self.layer_sizes) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [layer_size // 2], 1
                    )
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            p.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        return torch.cat(p, 1).sum(-1)

    def __str__(self):
        return super().__str__()


class AttentionBlock(BaseBlock):
    def __init__(self, block_config, feat_dim):
        super(AttentionBlock, self).__init__(block_config, feat_dim)
        self.block_option = self.block_config.get_attention_block()
        self._init_basic_block_params()
        self._build_arc()

    def _build_arc(self):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return
        self.emb_config = self.block_option.emb_config
        self.att_embed_dim = self.block_option.att_embed_dim
        self.num_of_heads = self.block_option.num_of_heads
        self.num_of_layers = self.block_option.num_of_layers
        self.use_res = self.block_option.use_res
        self.use_batchnorm = self.block_option.batchnorm
        self._dropout_p = self.block_option.dropout_prob

        self._refine_emb_arc()

        # set embeding layers
        self.feat_emb = create_emb_converter(
            self.num_dense_feat,
            self.feat_sparse_id,
            self.feat_sparse_dim,
            self.emb_config.comm_embed_dim,
            self.num_dense_as_sparse_feat,
        )

        # set attention params
        self.query_layers, self.key_layers, self.value_layers, self.res_layers, self.bn_layers = create_transformer(
            self.emb_config.comm_embed_dim,
            self.att_embed_dim,
            self.num_of_heads,
            self.num_of_layers,
            self.use_res,
            self.use_batchnorm,
        )

    def dim_config(self, feat_dim):
        if self.num_sparse_feat + self.num_dense_feat != 0:
            if self.num_dense_feat > 0:
                feat_dim["sparse"][self.block_id] = [
                    self.att_embed_dim * self.num_of_heads
                ] * (self.num_sparse_feat + 1)
            else:
                feat_dim["sparse"][self.block_id] = [
                    self.att_embed_dim * self.num_of_heads
                ] * self.num_sparse_feat
        return feat_dim

    def forward(self, feat_dict):
        if self.num_sparse_feat + self.num_dense_feat == 0:
            return feat_dict
        # extract dense features based on id
        extracted_feat_dict = {
            "dense": extract_dense_feat(feat_dict["dense"], self.feat_dense_id),
            "sparse": feat_dict["sparse"],
        }

        if self.num_dense_as_sparse_feat > 0:
            extracted_feat_dict["dense_as_sparse"] = (
                feat_dict["dense"][0]
                if self.dense_as_sparse_id == [-1]
                else feat_dict["dense"][0][:, self.dense_as_sparse_id]
            )

        # get feature matrix X0
        feat = convert_to_emb(
            extracted_feat_dict,
            self.feat_emb,
            self.num_dense_feat,
            self.feat_sparse_id,
            self.emb_config.comm_embed_dim,
            self.num_dense_as_sparse_feat,
        )
        if feat.dim() != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (feat.dim())
            )
        p = self.transformer(feat)
        feat_dict["sparse"][self.block_id] = {
            feat_id: p[:, feat_id] for feat_id in range(p.shape[1])  # 1 for dense
        }
        return feat_dict

    def transformer(self, feat):
        attention = feat
        for l in range(self.num_of_layers):
            Q = F.relu(self.query_layers[l](attention))
            K = F.relu(self.key_layers[l](attention))
            V = F.relu(self.value_layers[l](attention))
            if self.use_res:
                V_res = F.relu(self.res_layers[l](attention))

            # Split and concat
            Q_ = torch.cat(Q.split(split_size=self.att_embed_dim, dim=2), dim=0)
            K_ = torch.cat(K.split(split_size=self.att_embed_dim, dim=2), dim=0)
            V_ = torch.cat(V.split(split_size=self.att_embed_dim, dim=2), dim=0)

            # calculate QK^T
            weights = torch.matmul(Q_, K_.transpose(1, 2))
            # normalize with sqrt(dk)
            weights = weights / np.sqrt(self.att_embed_dim)

            # put it to softmax
            weights = F.softmax(weights, dim=-1)

            # apply dropout
            weights = F.dropout(weights, self._dropout_p)

            # multiply it with V
            attention = torch.matmul(weights, V_)
            # convert attention back to its input original size
            restore_chunk_size = int(attention.size(0) / self.num_of_heads)
            attention = torch.cat(
                attention.split(split_size=restore_chunk_size, dim=0), dim=2
            )

            # residual connection
            if self.use_res:
                attention += V_res

            # TODO: do we need this?
            attention = F.relu(attention)

            # apply batch normalization
            if self.use_batchnorm:
                attention = self.bn_layers[l](attention.transpose(1, 2)).transpose(1, 2)

        return attention

    def __str__(self):
        return super().__str__()
