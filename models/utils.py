# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
import torch.nn as nn

from config import ttypes as config


logger = logging.getLogger(__name__)


def apply_emb(feats, emb_dict, sparse_hash_size):
    ly = []
    for name, E in emb_dict.items():
        if name not in feats:
            raise ValueError("feature {} missing from input! ".format(name))
        val = feats[name]
        hash_size = sparse_hash_size[name]
        V = E(input=torch.remainder(val["data"], hash_size), offsets=val["offsets"])
        ly.append(V)
    return ly


def create_mlp(ln, ly_act=False):
    ln = list(ln)
    layers = nn.ModuleList()
    for i in range(1, len(ln) - 1):
        layers.append(nn.Linear(int(ln[i - 1]), int(ln[i]), bias=True))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(ln[-2]), int(ln[-1]), bias=True))
    if ly_act:
        layers.append(nn.ReLU())
    return torch.nn.Sequential(*layers)


def create_emb(sparse_feature, comm_embed_dim):
    embed_dim = (
        sparse_feature.embed_dim if sparse_feature.embed_dim > 0 else comm_embed_dim
    )
    hash_size = sparse_feature.hash_size
    if sparse_feature.pooling.getType() == config.PoolingConfig.SUM:
        mode = "sum"
    elif sparse_feature.pooling.getType() == config.PoolingConfig.AVG:
        mode = "mean"
    else:
        raise ValueError(
            "Unknown pooling option: {}".format(sparse_feature.pooling.getType())
        )
    # return nn.EmbeddingBag(hash_size, embed_dim, sparse=True, mode=mode)
    a = nn.EmbeddingBag(hash_size, embed_dim, sparse=True, mode=mode)
    nn.init.normal_(a.weight, 0, 0.01)
    return a


def create_emb_dict(sparse_feature_options):
    comm_embed_dim = sparse_feature_options.embed_dim
    return nn.ModuleDict(
        {
            item.name: create_emb(sparse_feature=item, comm_embed_dim=comm_embed_dim)
            for item in sparse_feature_options.features
        }
    )


def create_optim(params, optim_config):
    if optim_config.getType() == config.OptimConfig.SGD:
        opt_config = optim_config.get_sgd()
        return torch.optim.SGD(
            params,
            lr=opt_config.lr,
            momentum=opt_config.momentum,
            dampening=opt_config.dampening,
            weight_decay=opt_config.weight_decay,
            nesterov=opt_config.nesterov,
        )
    elif optim_config.getType() == config.OptimConfig.ADAGRAD:
        opt_config = optim_config.get_adagrad()
        return torch.optim.Adagrad(
            params,
            lr=opt_config.lr,
            lr_decay=opt_config.lr_decay,
            weight_decay=opt_config.weight_decay,
            initial_accumulator_value=opt_config.initial_accumulator_value,
        )
    elif optim_config.getType() == config.OptimConfig.SPARSE_ADAM:
        opt_config = optim_config.get_sparse_adam()
        return torch.optim.SparseAdam(
            params,
            lr=opt_config.lr,
            betas=(opt_config.betas0, opt_config.betas1),
            eps=opt_config.eps,
        )
    elif optim_config.getType() == config.OptimConfig.ADAM:
        opt_config = optim_config.get_adam()
        return torch.optim.Adam(
            params,
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            amsgrad=opt_config.amsgrad,
            betas=(opt_config.betas0, opt_config.betas1),
            eps=opt_config.eps,
        )
    elif optim_config.getType() == config.OptimConfig.RMSPROP:
        opt_config = optim_config.get_rmsprop()
        return torch.optim.RMSprop(
            params,
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            alpha=opt_config.alpha,
            momentum=opt_config.momentum,
            centered=opt_config.centered,
            eps=opt_config.eps,
        )
    else:
        raise ValueError("unknown optimizer type: {}".format(optim_config))


class Optimizers(object):
    def __init__(self, optimizers=None, named_optimizers=None):
        self.optimizers = [] if optimizers is None else optimizers
        self.named_optimizers = {} if named_optimizers is None else named_optimizers

    def add(self, optimizer, name=None):
        if name is None:
            self.optimizers.append(optimizer)
        else:
            assert (
                name not in self.named_optimizers
            ), "optimizer for {} already exist!".format(name)
            self.named_optimizers[name] = optimizer

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        for _, optimizer in self.named_optimizers.items():
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
        for _, optimizer in self.named_optimizers.items():
            optimizer.step()


# Assumes that embedding params have [sparse_name_key] (default "emb_dict")
# in their name. It is true for embeddings created via
#   self.emb_dict = create_emb_dict(self.sparse_feature_options)
def create_optimizers_for_dense(
    optimizers, named_parameters, dense_optim_config, sparse_name_key="emb_dict"
):
    params = [param for name, param in named_parameters if sparse_name_key not in name]
    logger.info(
        "Creating optim for non-embedding params with config: "
        "{}.".format(dense_optim_config)
    )
    logger.info(
        "Creating optim for non-embedding params list:"
        ", ".join([name for name, _ in named_parameters if sparse_name_key not in name])
    )
    optimizers.add(
        create_optim(params=params, optim_config=dense_optim_config), name="dense"
    )


def create_optimizers_for_embed(optimizers, emb_dict, sparse_feature_options):
    sparse_optim_config = sparse_feature_options.optim
    for item in sparse_feature_options.features:
        name = item.name
        item_optim_config = sparse_optim_config if item.optim is None else item.optim
        logger.info(
            "Creating optim for {} with config: {}".format(name, item_optim_config)
        )
        optimizers.add(
            create_optim(
                params=emb_dict[name].parameters(), optim_config=item_optim_config
            ),
            name=name,
        )
