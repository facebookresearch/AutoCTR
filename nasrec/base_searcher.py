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

from block_config import ttypes as b_config
from config import ttypes as config


logger = logging.getLogger(__name__)


class BaseSearcher(nn.Module):
    def __init__(self, searcher_config, feature_config):
        super(BaseSearcher, self).__init__()

        # for serilization purpose
        self.searcher_config = deepcopy(searcher_config)
        self.feature_config = deepcopy(feature_config)

        self.dense_feature_options = self.feature_config.dense
        self.sparse_feature_options = self.feature_config.sparse

        self.num_dense_feat = len(self.dense_feature_options.features)
        self.num_sparse_feat = len(self.sparse_feature_options.features)

    def _set_micro_space_from_config(self):
        # get micro space type list
        self.micro_space_types = [
            space_type.getType()
            for space_type in self.controller_option.micro_space_types
        ]
        # get feature processig type list
        self.feature_processing_type = [
            processing_type.getType()
            for processing_type in self.controller_option.feature_processing_type
        ]
        # set up corresponding micro space
        for space_type in self.controller_option.micro_space_types:
            if space_type.getType() == config.MicroSearchSpaceType.MICRO_MLP:
                self.micro_mlp_option = space_type.get_micro_mlp()
            elif space_type.getType() == config.MicroSearchSpaceType.MICRO_CIN:
                self.micro_cin_option = space_type.get_micro_cin()
                if len(self.micro_cin_option.arc) == 0:
                    self.micro_cin_option.arc = [128]
                if len(self.micro_cin_option.num_of_layers) == 0:
                    self.micro_cin_option.num_of_layers = [1]
            elif space_type.getType() == config.MicroSearchSpaceType.MICRO_ATTENTION:
                self.micro_attention_option = space_type.get_micro_attention()
                if len(self.micro_attention_option.num_of_layers) == 0:
                    self.micro_attention_option.num_of_layers = [1]
                if len(self.micro_attention_option.num_of_heads) == 0:
                    self.micro_attention_option.num_of_heads = [2]
                if len(self.micro_attention_option.att_embed_dim) == 0:
                    self.micro_attention_option.att_embed_dim = [10]
                if len(self.micro_attention_option.dropout_prob) == 0:
                    self.micro_attention_option.dropout_prob = [0.0]

    def _init_base_searcher_params(self):

        # get micro search space configurations
        self._set_micro_space_from_config()

        # constraint search space
        if (
            self.controller_option.macro_space_type
            == config.MacroSearchSpaceType.INPUT_GROUP
        ):
            self.num_dense_feat = 1
            self.num_sparse_feat = 1

        # length of the DAG to be searched (exclude the final clf layer)
        self.num_blocks = self.controller_option.max_num_block
        # block_types to be searched
        self.block_types = list(set(self.controller_option.block_types))
        self.num_block_type = len(self.block_types)
        if self.num_block_type == 0:
            raise ValueError("Should provide at least one block type to be searched.")

        # construct dictionaries to map between int and block types
        self.type_int_dict = {
            self.block_types[i]: i for i in range(self.num_block_type)
        }
        self.int_type_dict = {
            i: self.block_types[i] for i in range(self.num_block_type)
        }

        # all tokens to be searched
        self.num_tokens = {
            "block_type": self.num_block_type,
            "dense_feat": self.num_dense_feat,
            "sparse_feat": self.num_sparse_feat,
            "skip_connect": self.num_blocks,
        }
        self.token_names = ["block_type", "dense_feat", "sparse_feat", "skip_connect"]
        if (
            self.controller_option.macro_space_type
            == config.MacroSearchSpaceType.INPUT_ELASTIC_PRIOR
        ):
            # constraint search space with smooth learnable priors
            self.num_tokens["elastic_prior"] = 2
            self.token_names.append("elastic_prior")

        self.num_total_tokens = sum(v for _, v in self.num_tokens.items())

        if config.MicroSearchSpaceType.MICRO_MLP in self.micro_space_types:
            if (
                b_config.ExtendedBlockType.MLP_DENSE
                in self.controller_option.block_types
            ):
                self.num_tokens["mlp_dense"] = len(self.micro_mlp_option.arc)
                self.token_names.append("mlp_dense")
                self.num_total_tokens += 1
            if b_config.ExtendedBlockType.MLP_EMB in self.controller_option.block_types:
                self.num_tokens["mlp_emb"] = len(self.micro_mlp_option.arc)
                self.token_names.append("mlp_emb")
                self.num_total_tokens += 1

        if config.MicroSearchSpaceType.MICRO_CIN in self.micro_space_types:
            if b_config.ExtendedBlockType.CIN in self.controller_option.block_types:
                self.num_tokens["cin"] = len(self.micro_cin_option.arc) + len(
                    self.micro_cin_option.num_of_layers
                )
                self.token_names.append("cin")
                self.num_total_tokens += 1 if len(self.micro_cin_option.arc) > 0 else 0
                self.num_total_tokens += (
                    1 if len(self.micro_cin_option.num_of_layers) > 0 else 0
                )

        if config.MicroSearchSpaceType.MICRO_ATTENTION in self.micro_space_types:
            if (
                b_config.ExtendedBlockType.ATTENTION
                in self.controller_option.block_types
            ):
                self.att_num_tokens = {
                    "head": len(self.micro_attention_option.num_of_heads),
                    "layer": len(self.micro_attention_option.num_of_layers),
                    "emb": len(self.micro_attention_option.att_embed_dim),
                    "drop": len(self.micro_attention_option.dropout_prob),
                }
                self.num_tokens["attention"] = sum(
                    v for _, v in self.att_num_tokens.items()
                )

                self.token_names.append("attention")
                for _, v in self.att_num_tokens.items():
                    self.num_total_tokens += 1 if v != 0 else 0

    def _build_arc(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def random_sample(self):
        vec_configs, vecs = [], []
        for b_id in range(self.num_blocks):
            # macro random search
            block_type_vec = np.random.multinomial(
                1, [1.0 / self.num_block_type] * self.num_block_type
            )
            block_type_id = np.argmax(block_type_vec)
            dense_feat_vec = np.random.binomial(1, 0.5, self.num_dense_feat)
            sparse_feat_vec = np.random.binomial(1, 0.5, self.num_sparse_feat)
            skip_connection_vec = np.random.binomial(1, 0.5, self.num_blocks)
            skip_connection_vec[b_id:] = 0  # cannot connect with later block
            vec_config = {
                "block_type": block_type_id,
                "dense_feat": dense_feat_vec,
                "sparse_feat": sparse_feat_vec,
                "skip_connect": skip_connection_vec,
            }

            # micro random search
            mlp_dense_vec, mlp_emb_vec, cin_vec, att_vec = (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )
            if config.MicroSearchSpaceType.MICRO_MLP in self.micro_space_types:
                if (
                    b_config.ExtendedBlockType.MLP_DENSE
                    in self.controller_option.block_types
                ):
                    mlp_dense_vec = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / len(self.micro_mlp_option.arc)]
                            * len(self.micro_mlp_option.arc),
                        )
                    )
                    vec_config["mlp_dense"] = mlp_dense_vec
                    mlp_dense_vec = np.array([mlp_dense_vec])
                if (
                    b_config.ExtendedBlockType.MLP_EMB
                    in self.controller_option.block_types
                ):
                    mlp_emb_vec = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / len(self.micro_mlp_option.arc)]
                            * len(self.micro_mlp_option.arc),
                        )
                    )
                    vec_config["mlp_emb"] = mlp_emb_vec
                    mlp_emb_vec = np.array([mlp_emb_vec])
            if config.MicroSearchSpaceType.MICRO_CIN in self.micro_space_types:
                if b_config.ExtendedBlockType.CIN in self.controller_option.block_types:
                    cin_width = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / len(self.micro_cin_option.arc)]
                            * len(self.micro_cin_option.arc),
                        )
                    )
                    cin_depth = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / len(self.micro_cin_option.num_of_layers)]
                            * len(self.micro_cin_option.num_of_layers),
                        )
                    )
                    cin_vec = np.array([cin_width, cin_depth])
                    vec_config["cin"] = {"width": cin_width, "depth": cin_depth}
            if config.MicroSearchSpaceType.MICRO_ATTENTION in self.micro_space_types:
                if (
                    b_config.ExtendedBlockType.ATTENTION
                    in self.controller_option.block_types
                ):
                    att_head = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / self.att_num_tokens["head"]]
                            * self.att_num_tokens["head"],
                        )
                    )
                    att_layer = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / self.att_num_tokens["layer"]]
                            * self.att_num_tokens["layer"],
                        )
                    )
                    att_emb_dim = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / self.att_num_tokens["emb"]]
                            * self.att_num_tokens["emb"],
                        )
                    )
                    att_dropout_prob = np.argmax(
                        np.random.multinomial(
                            1,
                            [1.0 / self.att_num_tokens["drop"]]
                            * self.att_num_tokens["drop"],
                        )
                    )
                    att_vec = np.array(
                        [att_head, att_layer, att_emb_dim, att_dropout_prob]
                    )
                    vec_config["attention"] = {
                        "head": att_head,
                        "layer": att_layer,
                        "emb": att_emb_dim,
                        "drop": att_dropout_prob,
                    }
            block_vec = np.concatenate(
                [
                    block_type_vec,
                    dense_feat_vec,
                    sparse_feat_vec,
                    skip_connection_vec,
                    mlp_dense_vec,
                    mlp_emb_vec,
                    cin_vec,
                    att_vec,
                ]
            )
            vecs.append(block_vec)
            vec_configs.append(vec_config)

        # cat the config of a architecture to one vector
        return vecs, vec_configs

    def block_type_to_int(self, block_config):
        if block_config.getType() == b_config.BlockConfig.MLP_BLOCK:
            block_option = block_config.get_mlp_block()
            key = (
                b_config.ExtendedBlockType.MLP_DENSE
                if block_option.type.getType() == b_config.BlockType.DENSE
                else b_config.ExtendedBlockType.MLP_EMB
            )
        elif block_config.getType() == b_config.BlockConfig.CROSSNET_BLOCK:
            block_option = block_config.get_crossnet_block()
            key = b_config.ExtendedBlockType.CROSSNET
        elif block_config.getType() == b_config.BlockConfig.FM_BLOCK:
            block_option = block_config.get_fm_block()
            key = (
                b_config.ExtendedBlockType.FM_DENSE
                if block_option.type.getType() == b_config.BlockType.DENSE
                else b_config.ExtendedBlockType.FM_EMB
            )
        elif block_config.getType() == b_config.BlockConfig.DOTPROCESSOR_BLOCK:
            block_option = block_config.get_dotprocessor_block()
            key = (
                b_config.ExtendedBlockType.DOTPROCESSOR_DENSE
                if block_option.type.getType() == b_config.BlockType.DENSE
                else b_config.ExtendedBlockType.DOTPROCESSOR_EMB
            )
        elif block_config.getType() == b_config.BlockConfig.CAT_BLOCK:
            block_option = block_config.get_cat_block()
            key = (
                b_config.ExtendedBlockType.CAT_DENSE
                if block_option.type.getType() == b_config.BlockType.DENSE
                else b_config.ExtendedBlockType.CAT_EMB
            )
        elif block_config.getType() == b_config.BlockConfig.CIN:
            block_option = block_config.get_cin_block()
            key = b_config.ExtendedBlockType.CIN
        elif block_config.getType() == b_config.BlockConfig.ATTENTION:
            block_option = block_config.get_attention_block()
            key = b_config.ExtendedBlockType.ATTENTION
        return self.type_int_dict[key], block_option

    def vecs_to_model_config(self, vecs):
        block_configs = []
        for block_id, vec in enumerate(vecs):
            block_configs.append(self.vec_to_block_config(vec, block_id + 1))
        return block_configs

    def vec_to_block_config(self, vec, block_id):
        """convert a controller vector to block_config
        """
        # split a vector and convert the corresponding part to the id format
        block_type_id = (
            vec["block_type"].numpy()[0]
            if type(vec["block_type"]) is torch.Tensor
            else vec["block_type"]
        )
        input_dense = vec["dense_feat"]
        input_sparse = vec["sparse_feat"]
        skip_connection = vec["skip_connect"]

        if (
            self.controller_option.macro_space_type
            == config.MacroSearchSpaceType.INPUT_GROUP
        ):
            input_dense_id = [-1] if input_dense == 1 else []
            input_sparse_id = [-1] if input_sparse == 1 else []
        else:
            input_dense_id = [i for i, e in enumerate(input_dense) if e == 1]
            input_sparse_id = [i for i, e in enumerate(input_sparse) if e == 1]
        skip_connection_id = [
            i + 1 for i, e in enumerate(skip_connection) if e == 1 and i + 1 < block_id
        ]

        dense_as_sparse = (
            True
            if config.FeatureProcessingType.IDASP in self.feature_processing_type
            else False
        )

        # construct input config
        # orignal input features
        input_feat_config = [
            b_config.FeatSelectionConfig(
                block_id=0, dense=input_dense_id, sparse=input_sparse_id
            )
        ]
        # input from other blocks' outputs
        input_feat_config += [
            b_config.FeatSelectionConfig(block_id=id, dense=[-1], sparse=[-1])
            for id in skip_connection_id
        ]

        comm_embed_dim = self.sparse_feature_options.embed_dim

        block_type = self.int_type_dict[block_type_id]
        if block_type == b_config.ExtendedBlockType.CROSSNET:
            block_config = b_config.BlockConfig(
                crossnet_block=b_config.CrossNetBlockConfig(
                    name="CrossNetBlocks",
                    block_id=block_id,
                    num_of_layers=1,
                    input_feat_config=input_feat_config,
                    cross_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.ATTENTION:

            head, layer, emb, drop = (
                (
                    self.micro_attention_option.num_of_heads[vec["attention"]["head"]],
                    self.micro_attention_option.num_of_layers[
                        vec["attention"]["layer"]
                    ],
                    self.micro_attention_option.att_embed_dim[vec["attention"]["emb"]],
                    self.micro_attention_option.dropout_prob[vec["attention"]["drop"]],
                )
                if "attention" in vec
                else (2, 1, 10, 0.0)
            )
            block_config = b_config.BlockConfig(
                attention_block=b_config.AttentionBlockConfig(
                    name="AttentionBlock",
                    block_id=block_id,
                    input_feat_config=input_feat_config,
                    emb_config=b_config.EmbedBlockType(
                        comm_embed_dim=comm_embed_dim, dense_as_sparse=dense_as_sparse
                    ),
                    att_embed_dim=emb,
                    num_of_heads=head,
                    num_of_layers=layer,
                    dropout_prob=drop,
                    use_res=True,
                    batchnorm=False,
                )
            )
        elif block_type == b_config.ExtendedBlockType.CIN:
            arc = (
                [self.micro_cin_option.arc[vec["cin"]["width"]]]
                * self.micro_cin_option.num_of_layers[vec["cin"]["depth"]]
                if "cin" in vec
                else [128]
            )
            block_config = b_config.BlockConfig(
                cin_block=b_config.CINBlockConfig(
                    name="CINBlock",
                    block_id=block_id,
                    emb_config=b_config.EmbedBlockType(
                        comm_embed_dim=comm_embed_dim, dense_as_sparse=dense_as_sparse
                    ),
                    arc=arc,
                    split_half=True,
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.MLP_DENSE:
            arc = (
                self.micro_mlp_option.arc[vec["mlp_dense"]]
                if "mlp_dense" in vec
                else 128
            )
            block_config = b_config.BlockConfig(
                mlp_block=b_config.MLPBlockConfig(
                    name="MLPBlock",
                    block_id=block_id,
                    arc=[arc],
                    type=b_config.BlockType(dense=b_config.DenseBlockType()),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.MLP_EMB:
            arc = self.micro_mlp_option.arc[vec["mlp_emb"]] if "mlp_emb" in vec else 128
            block_config = b_config.BlockConfig(
                mlp_block=b_config.MLPBlockConfig(
                    name="MLPBlock",
                    block_id=block_id,
                    arc=[arc],
                    type=b_config.BlockType(
                        emb=b_config.EmbedBlockType(
                            comm_embed_dim=comm_embed_dim,
                            dense_as_sparse=dense_as_sparse,
                        )
                    ),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.FM_DENSE:
            block_config = b_config.BlockConfig(
                fm_block=b_config.FMBlockConfig(
                    name="FMBlock",
                    block_id=block_id,
                    type=b_config.BlockType(dense=b_config.DenseBlockType()),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.FM_EMB:
            block_config = b_config.BlockConfig(
                fm_block=b_config.FMBlockConfig(
                    name="FMBlock",
                    block_id=block_id,
                    type=b_config.BlockType(
                        emb=b_config.EmbedBlockType(
                            comm_embed_dim=comm_embed_dim,
                            dense_as_sparse=dense_as_sparse,
                        )
                    ),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.DOTPROCESSOR_DENSE:
            block_config = b_config.BlockConfig(
                dotprocessor_block=b_config.DotProcessorBlockConfig(
                    name="DotProcessorBlock",
                    block_id=block_id,
                    type=b_config.BlockType(dense=b_config.DenseBlockType()),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.DOTPROCESSOR_EMB:
            block_config = b_config.BlockConfig(
                dotprocessor_block=b_config.DotProcessorBlockConfig(
                    name="DotProcessorBlock",
                    block_id=block_id,
                    type=b_config.BlockType(
                        emb=b_config.EmbedBlockType(
                            comm_embed_dim=comm_embed_dim,
                            dense_as_sparse=dense_as_sparse,
                        )
                    ),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.CAT_DENSE:
            block_config = b_config.BlockConfig(
                cat_block=b_config.CatBlockConfig(
                    name="CatBlock",
                    block_id=block_id,
                    type=b_config.BlockType(dense=b_config.DenseBlockType()),
                    input_feat_config=input_feat_config,
                )
            )
        elif block_type == b_config.ExtendedBlockType.CAT_EMB:
            block_config = b_config.BlockConfig(
                cat_block=b_config.CatBlockConfig(
                    name="CatBlock",
                    block_id=block_id,
                    type=b_config.BlockType(
                        emb=b_config.EmbedBlockType(
                            comm_embed_dim=comm_embed_dim,
                            dense_as_sparse=dense_as_sparse,
                        )
                    ),
                    input_feat_config=input_feat_config,
                )
            )
        return block_config

    def dicts_to_vecs(self, dicts):
        vecs = []
        for block in dicts:
            for token_name in self.num_tokens:
                if token_name in ["block_type"]:
                    tmp_vec = np.zeros([self.num_tokens[token_name]])
                    tmp_vec[block[token_name]] = 1.0
                    vecs.append(tmp_vec)
                elif token_name in ["mlp_dense", "mlp_emb"]:
                    tmp_vec = np.array([block[token_name]])
                    vecs.append(tmp_vec)
                elif token_name == "cin":
                    tmp_vec = np.array([block["cin"]["width"], block["cin"]["depth"]])
                    vecs.append(tmp_vec)
                elif token_name == "attention":
                    tmp_vec = np.array(
                        [
                            block["attention"]["head"],
                            block["attention"]["layer"],
                            block["attention"]["emb"],
                            block["attention"]["drop"],
                        ]
                    )
                    vecs.append(tmp_vec)
                else:
                    vecs.append(block[token_name])
        return vecs

    def _action_equal(self, action1, action2):
        return (
            action1 == action2
            if type(action1) == dict
            else np.array_equal(action1, action2)
        )

    def mutate_arc(self, parent):
        child = deepcopy(parent)
        # 1. choose block to mutate
        block_id = np.random.choice(self.num_blocks, 1)[0]
        # 2. choose one token of a block to mutate (e.g., block_type, dense_feat)
        token_name = np.random.choice(self.token_names, 1)[0]
        while token_name == "skip_connect" and block_id == 0:
            block_id = np.random.choice(self.num_blocks, 1)[0]
            token_name = np.random.choice(self.token_names, 1)[0]
        while (
            token_name == "cin"
            and len(self.micro_cin_option.arc) == 1
            and len(self.micro_cin_option.num_of_layers) == 1
        ) or (
            token_name == "attention"
            and self.att_num_tokens["head"] == 1
            and self.att_num_tokens["layer"] == 1
            and self.att_num_tokens["emb"] == 1
            and self.att_num_tokens["drop"] == 1
        ):
            token_name = np.random.choice(self.token_names, 1)[0]
        # 3. mutate the corresponding token
        new_action = child[block_id][token_name]
        while self._action_equal(new_action, child[block_id][token_name]):
            if token_name in ["block_type", "mlp_dense", "mlp_emb"]:
                new_action_vec = np.random.multinomial(
                    1, [1.0 / self.num_tokens[token_name]] * self.num_tokens[token_name]
                )
                new_action = np.argmax(new_action_vec)
            elif token_name == "cin":
                cin_width = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / len(self.micro_cin_option.arc)]
                        * len(self.micro_cin_option.arc),
                    )
                )
                cin_depth = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / len(self.micro_cin_option.num_of_layers)]
                        * len(self.micro_cin_option.num_of_layers),
                    )
                )
                new_action = {"width": cin_width, "depth": cin_depth}
            elif token_name == "attention":
                head = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / self.att_num_tokens["head"]]
                        * self.att_num_tokens["head"],
                    )
                )
                layer = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / self.att_num_tokens["layer"]]
                        * self.att_num_tokens["layer"],
                    )
                )
                emb = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / self.att_num_tokens["emb"]] * self.att_num_tokens["emb"],
                    )
                )
                drop = np.argmax(
                    np.random.multinomial(
                        1,
                        [1.0 / self.att_num_tokens["drop"]]
                        * self.att_num_tokens["drop"],
                    )
                )
                new_action = {"head": head, "layer": layer, "emb": emb, "drop": drop}
            else:
                new_action = np.random.binomial(1, 0.5, self.num_tokens[token_name])
        child[block_id][token_name] = new_action
        vecs = self.dicts_to_vecs(child)
        return vecs, child
