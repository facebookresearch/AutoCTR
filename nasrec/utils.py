# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import torch
import torch.nn as nn
from graphviz import Digraph


logger = logging.getLogger(__name__)


def reward_normalization(rewards, alpha=3, bias=0.5):
    return rewards
    # return 0.5 * np.tanh((rewards - bias) * alpha) + 0.5


def clean_feat_id(feat_ids, feat_dim, feat_type):
    """check and modify feat_ids, remove nonexist features and empty block
    Args:
        feat_ids: dictionary of {block:feat_ids} to be cleaned
        feat_dim: dictionary of {block:feat_dim} used to clean feat_ids
        feat_type: a string indicating feature type (i.e., "dense" or "sprase")
    """
    tmp = {
        k: [
            feat_id
            for feat_id in set(feat_ids[k])
            if feat_id < (feat_dim[k][0] if feat_type is "dense" else len(feat_dim[k]))
        ]
        for k in set(feat_ids).intersection(set(feat_dim))
    }
    # remove empty and sorted
    return {k: sorted(v) for k, v in tmp.items() if v}


def create_emb_converter(
    num_dense_feat, feat_sparse_id, feat_sparse_dim, comm_embed_dim, num_dense_as_sp=0
):
    # set embedding layers
    feat_emb = nn.ModuleDict()

    # set dense emb layer
    if num_dense_feat > 0:
        feat_emb["dense"] = (
            nn.Linear(num_dense_feat, comm_embed_dim, bias=True)
            if num_dense_feat != comm_embed_dim
            else nn.Identity()
        )
    if num_dense_as_sp > 0:
        feat_emb["dense_as_sparse"] = nn.Embedding(num_dense_as_sp, comm_embed_dim)

    # set sparse emb layer
    feat_emb["sparse"] = nn.ModuleDict()
    sparse_in_dim = get_sparse_feat_dim(feat_sparse_id, feat_sparse_dim)
    for block in feat_sparse_id:
        feat_emb["sparse"][str(block)] = nn.ModuleDict()
        if feat_sparse_id[block] == [-1]:
            for feat_id in range(len(sparse_in_dim[block])):
                feat_emb["sparse"][str(block)][str(feat_id)] = (
                    nn.Linear(sparse_in_dim[block][feat_id], comm_embed_dim, bias=True)
                    if sparse_in_dim[block][feat_id] != comm_embed_dim
                    else nn.Identity()
                )
        else:
            for feat_id in feat_sparse_id[block]:
                feat_emb["sparse"][str(block)][str(feat_id)] = (
                    nn.Linear(sparse_in_dim[block][feat_id], comm_embed_dim, bias=True)
                    if sparse_in_dim[block][feat_id] != comm_embed_dim
                    else nn.Identity()
                )
    return feat_emb


def convert_to_emb(
    feat_dict,
    feat_emb_layers,
    num_dense_feat,
    feat_sparse_id,
    comm_embed_dim,
    num_dense_as_sp=0,
):
    """
    :param num_dense_as_sp: # of input dense features to be treated as sparse features
    """
    # embedding all features into the same length and concatenate them into a matrix
    # dense
    feat = [] if num_dense_feat <= 0 else [feat_emb_layers["dense"](feat_dict["dense"])]

    # sparse
    sp_feats = []
    for block in feat_sparse_id:
        if feat_sparse_id[block] == [-1]:
            for feat_id, sp in feat_dict["sparse"][block].items():
                emb = feat_emb_layers["sparse"][str(block)][str(feat_id)]
                sp = sp.to(dtype=torch.float)
                sp_feats.append(emb(sp))
        else:
            for feat_id in feat_sparse_id[block]:
                emb = feat_emb_layers["sparse"][str(block)][str(feat_id)]
                sp = feat_dict["sparse"][block][feat_id]
                sp = sp.to(dtype=torch.float)
                sp_feats.append(emb(sp))

    # dense_to_sparse
    if num_dense_as_sp > 0:
        emb_table = feat_emb_layers["dense_as_sparse"](
            torch.tensor(list(range(num_dense_as_sp)))
        )
        emb_table = emb_table.repeat([feat_dict["dense_as_sparse"].shape[0], 1, 1])
        dense_as_sp_feat = emb_table * feat_dict["dense_as_sparse"][:, :, None]

    # concatenation
    if feat + sp_feats:
        feat = torch.cat(feat + sp_feats, dim=1)
        batch_size = feat.shape[0]
        feat = feat.view((batch_size, -1, comm_embed_dim))
        if num_dense_as_sp > 0:
            feat = torch.cat([feat, dense_as_sp_feat], dim=1)
    else:
        feat = dense_as_sp_feat
    return feat


def cat_feats(feat_dict, feat_sparse_id):
    # concatenate all features into one row vector
    feat = [] if feat_dict["dense"].nelement() == 0 else [feat_dict["dense"]]
    sp_feats = []
    for block, feat_ids in feat_sparse_id.items():
        if feat_ids == [-1]:
            for feat_id in feat_dict["sparse"][block]:
                sp = feat_dict["sparse"][block][feat_id]
                sp = sp.to(dtype=torch.float)
                sp_feats.append(sp)
        else:
            for feat_id in feat_sparse_id[block]:
                sp = feat_dict["sparse"][block][feat_id]
                sp = sp.to(dtype=torch.float)
                sp_feats.append(sp)
    return torch.cat(feat + sp_feats, dim=1)


def extract_dense_feat(feat_dense_dict, feat_dense_id):
    # extract
    dense = []
    for block, feat_id in feat_dense_id.items():
        if feat_dense_dict[block].nelement() != 0:
            dense.append(
                feat_dense_dict[block]
                if feat_id == [-1]
                else feat_dense_dict[block][:, feat_id]
            )
    return torch.cat(dense, dim=1) if dense else torch.Tensor([])


def config_to_dict(feat_configs):
    feat_dense_id = {
        feat_config.block_id: feat_config.dense
        for feat_config in feat_configs
        if len(feat_config.dense)
    }
    feat_sparse_id = {
        feat_config.block_id: feat_config.sparse
        for feat_config in feat_configs
        if len(feat_config.sparse)
    }
    return feat_dense_id, feat_sparse_id


def get_sparse_feat_dim(feat_id_dict, feat_dim_dict):
    # get sparse feature dimension
    sparse_in_dim = {}
    for block, feat_ids in feat_id_dict.items():
        if feat_ids == [-1]:
            sparse_in_dim[block] = feat_dim_dict[block]
        else:
            sparse_in_dim[block] = {}
            for feat_id in feat_ids:
                sparse_in_dim[block][feat_id] = feat_dim_dict[block][feat_id]
    return sparse_in_dim


def get_sparse_feat_dim_num(feat_id_dict, feat_dim_dict):
    # get sparse feature dimension
    num_sparse_in_dim = 0
    for block, feat_ids in feat_id_dict.items():
        if feat_ids == [-1]:
            num_sparse_in_dim += sum(feat_dim_dict[block])
        else:
            for feat_id in feat_ids:
                num_sparse_in_dim += feat_dim_dict[block][feat_id]
    return num_sparse_in_dim


def create_crossnet(num_of_layers, num_input_feat):

    weight_w = torch.nn.ModuleList(
        [torch.nn.Linear(num_input_feat, 1, bias=False) for _ in range(num_of_layers)]
    )
    weight_b = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.zeros((num_input_feat,)))
            for _ in range(num_of_layers)
        ]
    )
    batchnorm = torch.nn.ModuleList(
        [nn.BatchNorm1d(num_input_feat, affine=False) for _ in range(num_of_layers)]
    )

    return weight_w, weight_b, batchnorm


def create_cin(layer_sizes, field_nums):
    conv_layers, bias_layers, activation_layers = (
        nn.ModuleList(),
        nn.ParameterList(),
        nn.ModuleList(),
    )
    for i, size in enumerate(layer_sizes):
        single_conv_layer = nn.Conv2d(
            in_channels=1, out_channels=size, kernel_size=(field_nums[i], field_nums[0])
        )
        conv_layers.append(single_conv_layer)
        bias_layers.append(
            nn.Parameter(torch.nn.init.normal_(torch.empty(size), mean=0.0, std=1e-6))
        )
        activation_layers.append(nn.ReLU())
    return conv_layers, bias_layers, activation_layers


def create_transformer(
    emb_dim, att_embed_dim, num_of_heads, num_of_layers, use_res, use_batchnorm
):
    w_query, w_key, w_value, w_res, bn = (
        nn.ModuleList(),
        nn.ModuleList(),
        nn.ModuleList(),
        nn.ModuleList(),
        nn.ModuleList(),
    )
    num_units = att_embed_dim * num_of_heads
    emb_dim = [emb_dim] + (num_of_layers - 1) * [num_units]
    for l in range(num_of_layers):
        w_query.append(nn.Linear(emb_dim[l], num_units, bias=True))
        w_key.append(nn.Linear(emb_dim[l], num_units, bias=True))
        w_value.append(nn.Linear(emb_dim[l], num_units, bias=True))
        if use_res:
            w_res.append(nn.Linear(emb_dim[l], num_units, bias=True))
        if use_batchnorm:
            bn.append(nn.BatchNorm1d(num_units))
    return w_query, w_key, w_value, w_res, bn


def nasnet_visual(nasrec_model):
    """ function to visualize the nasrec net model
    """
    dot = Digraph(comment="Graph", format="png")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("0_d", "Dense", color="red")
        s.node("0_s", "Sparse", color="red")

    block_name = []
    for i, block in enumerate(nasrec_model.blocks):
        block_name.append(block.__str__() + "Block")
        dot.node(
            str(i + 1), str(i + 1) + "_" + block_name[-1], shape="box", color="green"
        )

        dense = block.feat_dense_id
        sparse = block.feat_sparse_id
        skip_block_id = set(dense.keys()).union(set(sparse.keys()))
        cross_dense = []
        cross_sparse = []
        if block_name[-1] == "CrossNet":
            cross_dense = block.cross_feat_dense_id
            cross_sparse = block.cross_feat_sparse_id
            skip_block_id = skip_block_id.union(set(cross_dense.keys()))
            skip_block_id = skip_block_id.union(set(cross_sparse.keys()))

        for id in skip_block_id:
            if id == 0:
                if id in dense or (cross_dense and id in cross_dense):
                    dot.edge("0_d", str(i + 1))
                if id in sparse or (cross_sparse and id in cross_sparse):
                    dot.edge("0_s", str(i + 1))
            else:
                dot.edge(str(id), str(i + 1))
    return dot
