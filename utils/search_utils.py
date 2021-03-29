# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

sys.path.append('gen-py')

import argparse
import json

from block_config import ttypes as b_config
from config import ttypes as config


def get_args():
    parser = argparse.ArgumentParser(
        description="Neural Recommendation Model Searching Script for Kaggle Dataset"
    )
    # configs for final fit only
    parser.add_argument("--model-file", type=str, default="",
        help="a json file contain the model structure for final fit")
    parser.add_argument("--save-model", action="store_true", default=False, help="save model or not during the final fit process")


    # configs for search and final fit
    parser.add_argument("--data-file", type=str, default="", help="data for search or final fit")
    parser.add_argument("--data-set-name", type=str, default="", help="dataset name", choices=["criteo", "avazu", "kdd2012"])
    parser.add_argument("--log-freq", type=int, default=10, help="log freqency of model training (# of epochs)")
    parser.add_argument("--splits", type=str, default="0.8:0.1",
        help="split of train,val,test, e.g., 0.8:0.1 means 80% train, 10% val, 10% test")

    parser.add_argument("--batch-size", type=int, default=100, help="batch size for training each model")
    parser.add_argument("--hash-size", type=int, default=10000, help="hash size for the features")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate of each model")
    parser.add_argument("--nepochs", type=int, default=50, help="maximum epoch for training a model")
    parser.add_argument("--num-workers", type=int, default=4,
        help="number of workers (cpus) to preprocess data")

    parser.add_argument("--num-trainers", type=int, default=1,
        help="number of training for cpu training, currently this is abandoned and to be removed, we only support gpu training now")

    parser.add_argument("--repeat-checker-off", action="store_true", default=False, help="check and avoid repeating searching same architectures")

    parser.add_argument(
        "--save-model-path", type=str, default="", help="the file path to save the models during the search process"
    )

    parser.add_argument("--search-nepochs", type=int, default=3, help="number of search iterations")
    parser.add_argument(
        "--reward-type",
        default="logloss",
        type=str,
        choices=["logloss", "auc"],
        help="measurement for the search model to compare models"
    )
    parser.add_argument(
        "--searcher-type",
        default="random",
        type=str,
        choices=["random", "evo"],
        help="search algorithm"
    )
    parser.add_argument("--max-num-block", type=int, default=5, help="maximum number of blocks in each model in the search space")
    parser.add_argument(
        "--feature-processing-type", default="", type=str, choices=["idasp"], help="if we want to treat dense feature as sparse features"
    )

    # hyperparameters for proposed evo algorithm
    parser.add_argument("--population-size", type=int, default=3,
        help="size of the population, it also decides how many random initialization architectures we will do")
    parser.add_argument("--candidate-size", type=float, default=2,
        help="number of candidates to be picked from the population, the best one will be used to generate offsprings")
    parser.add_argument("--sampler-type", type=int, default=10, help="number of neigbors for each candidate")
    parser.add_argument("--historical-sample-path", type=str, default="", help="path for historical architectures to warm start the evo searcher")
    parser.add_argument("--historical-sample-num", type=int, default=0, help="number of historical architectures to warm start the evo searcher")
    parser.add_argument(
        "--survival-type", default="comb", type=str, choices=["age", "fit", "mix", "comb"],
        help="survival type, comb is multi-objective survival function, mix is a two-step survival function"
    )

    # search space config
    parser.add_argument(
        "--macro-space-type", type=int, default=config.MacroSearchSpaceType.INPUT_GROUP,
        help="search space for features, either group sparse/dense features or not, please check out the /if/config.thrift for more detail"
    )
    parser.add_argument(
        "--micro-space-types",
        default="close",
        type=str,
        choices=[
            "close",
            "micro_mlp",
        ],
        help="micro search space for blocks, currently only mlp have a micro space hyperparameter (units in each mlp layer), close means do not search mlp units",
    )

    # general search config
    parser.add_argument("--num-machines", type=int, default=1, help="number of GPUs to be used")

    parser.add_argument("--waiting-time", type=float, default=30,
        help="waiting time for checking if the current running models are complete, default: check every 30 seconds")

    parser.add_argument("--resume-file", type=str, default="", help="the file path to resume the search process")

    parser.add_argument("--fbl-kill-time", type=float, default=1800,
        help="time to kill a model during search, this is used to avoid some model crush and stuck during training")

    parser.add_argument("--numpy-seed", type=int, default=123, help="numpy seed")
    parser.add_argument("--torch-seed", type=int, default=4321, help="torch seed")

    parser.add_argument("--warm-start-emb", action="store_true", default=False,
        help="if we have a `.ckp` model weight to warm start the embeddings of the sparse features in each model")

    # gpu config
    parser.add_argument("--use-gpu", action="store_true", default=False, help="use gpu or not")

    parser.add_argument("--maxLoad", type=float, default=0.5,
        help="only load a model when the current used load of this gpu is lower than maxLoad")

    parser.add_argument("--maxMemory", type=float, default=0.5,
        help="only load a model when the current used memory of this gpu is lower than maxMemory")

    parser.add_argument("--save-batches", action="store_true", default=False,
        help="if we want to save the training data batches in the gpu memory, this will accelerate the speed")

    parser.add_argument("--save-val-batches", action="store_true", default=False,
        help="if we want to save the validation data batches in the gpu memory, this will accelerate the speed")

    parser.add_argument("--total-gpus", type=int, default=1, help="total number of gpus on the machine")
    parser.add_argument("--excludeID", type=str, default="", help="")

    args = parser.parse_args()

    if not args.save_model_path:
        args.save_model_path = os.path.join(os.getcwd(), "results")

    return args


def get_micro_space_types(args):
    micro_space_types = args.micro_space_types.replace(" ", "")
    micro_space_types = micro_space_types.split(",")
    micro_space_types = list(set(micro_space_types))
    micro_space_configs = []
    if "close" in micro_space_types:
        return [config.MicroSearchSpaceType(close=config.MicroClose())]
    elif "micro_mlp" in micro_space_types:
        micro_space_configs.append(
            config.MicroSearchSpaceType(
                micro_mlp=config.MicroMLPConfig(arc=[32, 64, 128, 256, 512, 1024])
            )
        )
    elif "micro_cin" in micro_space_types:
        micro_space_configs.append(
            config.MicroSearchSpaceType(
                micro_cin=config.MicroCINConfig(
                    arc=[64, 128, 256], num_of_layers=[1, 2, 3]
                )
            )
        )
    elif "micro_attention" in micro_space_types:
        micro_space_configs.append(
            config.MicroSearchSpaceType(
                micro_attention=config.MicroAttentionConfig(
                    num_of_layers=[1, 2, 3],
                    num_of_heads=[1, 2, 3],
                    att_embed_dim=[],
                    dropout_prob=[],
                )
            )
        )
    else:
        raise ValueError("Error micro space type.")
    return micro_space_configs


def get_feature_processing_type(args):
    feature_processing_type = args.feature_processing_type.replace(" ", "")
    feature_processing_type = feature_processing_type.split(",")
    feature_processing_type = list(set(feature_processing_type))
    feature_processing_configs = []
    if feature_processing_type != [""]:
        if "idasp" in feature_processing_type:
            feature_processing_configs.append(
                config.FeatureProcessingType(idasp=config.InputDenseAsSparse())
            )
        else:
            raise ValueError("Error micro space type.")
    return feature_processing_configs


def get_searcher_config(args):
    block_types = [
        b_config.ExtendedBlockType.MLP_DENSE,
        # b_config.ExtendedBlockType.MLP_EMB,
        # b_config.ExtendedBlockType.CROSSNET,
        # b_config.ExtendedBlockType.FM_DENSE,
        b_config.ExtendedBlockType.FM_EMB,
        # b_config.ExtendedBlockType.DOTPROCESSOR_DENSE,
        b_config.ExtendedBlockType.DOTPROCESSOR_EMB,
        # b_config.ExtendedBlockType.CAT_DENSE,
        # b_config.ExtendedBlockType.CAT_EMB,
        # b_config.ExtendedBlockType.CIN,
        # b_config.ExtendedBlockType.ATTENTION,
    ]
    if args.searcher_type == "random":
        searcher_config = config.SearcherConfig(
            random_searcher=config.RandomSearcherConfig(
                max_num_block=args.max_num_block,
                block_types=block_types,
                macro_space_type=args.macro_space_type,
                micro_space_types=get_micro_space_types(args),
                feature_processing_type=get_feature_processing_type(args),
            )
        )
    elif args.searcher_type == "evo":
        searcher_config = config.SearcherConfig(
            evolutionary_searcher=config.EvolutionarySearcherConfig(
                max_num_block=args.max_num_block,
                block_types=block_types,
                population_size=args.population_size,
                candidate_size=max(1, int(args.candidate_size)),
                macro_space_type=args.macro_space_type,
                micro_space_types=get_micro_space_types(args),
                feature_processing_type=get_feature_processing_type(args),
            )
        )
    return searcher_config


def get_trainer_config(args):
    fp = os.getcwd()
    if args.data_set_name == "criteo":
        input_summary = json.load(open(fp + "/utils/fblearner_template/criteo_search.json"))
    elif args.data_set_name == "avazu":
        input_summary = json.load(open(fp + "/utils/fblearner_template/avazu_search.json"))
    elif args.data_set_name == "kdd2012":
        input_summary = json.load(open(fp + "/utils/fblearner_template/kdd2012_search.json"))
    else:
        input_summary = json.load(open(fp + "/utils/fblearner_template/criteo_search.json"))

    return input_summary, args


def get_final_fit_trainer_config(args):
    fp = os.getcwd()
    if args.data_set_name == "criteo":
        input_summary = json.load(open(fp + "/utils/fblearner_template/criteo_transfer.json"))
    elif args.data_set_name == "avazu":
        input_summary = json.load(open(fp + "/utils/fblearner_template/avazu_transfer.json"))
    elif args.data_set_name == "kdd2012":
        input_summary = json.load(open(fp + "/utils/fblearner_template/kdd2012_transfer.json"))
    else:
        input_summary = json.load(open(fp + "/utils/fblearner_template/criteo_transfer.json"))

    return input_summary, args


def get_phenotype(args):

    filenames = [args.model_file]

    model_config_dicts = []
    for filename in filenames:
        with open(filename) as fp:
            model_config_dict = json.load(fp)
        fp.close()
        model_config_dicts.append(model_config_dict)

    return filenames, model_config_dicts
