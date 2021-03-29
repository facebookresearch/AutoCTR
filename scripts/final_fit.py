# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import sys

sys.path.append('gen-py')

import json
import logging
import os
import pickle
import time

import numpy as np
import torch

# os.system(f"mount -o remount,size={60*1024*1024*1024} /dev/shm")

from thrift.protocol import TSimpleJSONProtocol
from thrift.util import Serializer

from config import ttypes as config
from models.nas_modules import NASRecNet
from trainers.simple_final import train as simple_train
from utils.data import prepare_data
from utils.search_utils import get_args, get_final_fit_trainer_config, get_phenotype

from torch.multiprocessing import Pipe, Process, set_start_method

set_start_method('spawn', force=True)

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

import GPUtil

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

jfactory = TSimpleJSONProtocol.TSimpleJSONProtocolFactory()

THRESHOLD = -1 # -1 # 7500
VAL_THRESHOLD = -1

if __name__ == "__main__":
    # get arguments
    args = get_args()

    logger.warning("All Args: {}".format(args))
    # set seeds
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    excludeID = [int(id) for id in args.excludeID.split(",")] if args.excludeID else []

    # get model
    filenames, model_config_dicts = get_phenotype(args)

    # get trainer config
    input_summary, args = get_final_fit_trainer_config(args)

    # change dataset to small dataset
    input_summary["data_options"]["from_file"]["data_file"] = args.data_file
    input_summary["data_options"]["from_file"]["batch_size"] = args.batch_size

    # change train_options
    input_summary["train_options"]["nepochs"] = args.nepochs
    input_summary["train_options"]["logging_config"]["log_freq"] = 100000
    input_summary["train_options"]["logging_config"]["tb_log_freq"] = 100000

    # change performance_options
    input_summary["performance_options"]["num_readers"] = args.num_workers
    input_summary["performance_options"]["num_trainers"] = args.num_trainers
    input_summary["performance_options"]["use_gpu"] = args.use_gpu

    # change optimizer
    input_summary["feature_options"]["dense"]["optim"]["adam"]["lr"] = args.learning_rate
    input_summary["feature_options"]["sparse"]["optim"]["sparse_adam"]["lr"] = args.learning_rate

    # # change feature hashing size
    # for i, feature in enumerate(input_summary["feature_options"]["sparse"]["features"]):
    #     if feature["hash_size"] > args.hash_size:
    #         input_summary["feature_options"]["sparse"]["features"][i]["hash_size"] = args.hash_size

    # data_options
    splits = [float(p) for p in args.splits.split(":")]
    input_summary["data_options"]["from_file"]["splits"] = splits

    # extract feature config for searcher construction and trainer
    train_options = Serializer.deserialize(
        jfactory,
        json.dumps(input_summary["train_options"]),
        config.TrainConfig(),
    )

    # extract feature config for searcher construction and trainer
    feature_config = Serializer.deserialize(
        jfactory,
        json.dumps(input_summary["feature_options"]),
        config.FeatureConfig(),
    )

    data_options = Serializer.deserialize(
        jfactory,
        json.dumps(input_summary["data_options"]),
        config.DataConfig(),
    )

    performance_options = Serializer.deserialize(
        jfactory,
        json.dumps(input_summary["performance_options"]),
        config.PerformanceConfig(),
    )

    # for datasaving purpose
    batch_processor, train_dataloader, val_dataloader, eval_dataloader, \
    train_dataloader_batches, val_dataloader_batches, eval_dataloader_batches \
        = {}, {}, {}, {}, {}, {}, {}

    for id in range(args.total_gpus):
        if id not in excludeID:
            CUDA = 'cuda:' + str(id)
            if len(batch_processor) == 0:
                (
                    _,  # datasets
                    batch_processor[CUDA],
                    train_dataloader,
                    val_dataloader,
                    eval_dataloader,
                ) = prepare_data(data_options, performance_options, CUDA, pin_memory=False)
            else:
                (
                    _,  # datasets
                    batch_processor[CUDA],
                    _,
                    _,
                    _,  # eval_dataloader
                ) = prepare_data(data_options, performance_options, CUDA, pin_memory=True)
                train_dataloader = None
                val_dataloader = None
                eval_dataloader = None


            train_dataloader_batches[CUDA] = None
            val_dataloader_batches[CUDA] = None
            eval_dataloader_batches[CUDA] = None

            if args.save_batches:
                train_dataloader_batches[CUDA] = []
                if len(batch_processor) == 1:
                    for i_batch, sample_batched in enumerate(train_dataloader):
                        if i_batch % 100 == 0:
                            logger.warning("i_batch {}".format(i_batch))
                        train_dataloader_batches[CUDA].append(sample_batched)
                    mark = CUDA

                # if args.save_val_batches:
                val_dataloader_batches[CUDA] = []
                if len(batch_processor) == 1:
                    for i_batch, sample_batched in enumerate(val_dataloader):
                        if i_batch % 100 == 0:
                            logger.warning("i_batch {}".format(i_batch))
                        val_dataloader_batches[CUDA].append(sample_batched)
                    mark = CUDA

                # if args.save_val_batches:
                eval_dataloader_batches[CUDA] = []
                if len(batch_processor) == 1:
                    for i_batch, sample_batched in enumerate(eval_dataloader):
                        if i_batch % 100 == 0:
                            logger.warning("i_batch {}".format(i_batch))
                        eval_dataloader_batches[CUDA].append(sample_batched)
                    mark = CUDA

    if args.save_batches:
        for i_batch, sample_batched in enumerate(train_dataloader_batches[mark]):
            if i_batch % 100 == 0:
                logger.warning("process_first_cuda_i_batch {}".format(i_batch))
            if i_batch <= THRESHOLD:
                train_dataloader_batches[mark][i_batch] = batch_processor[mark](
                    mini_batch=sample_batched)
    if args.save_val_batches:
        for i_batch, sample_batched in enumerate(val_dataloader_batches[mark]):
            if i_batch % 100 == 0:
                logger.warning("process_first_cuda_i_batch {}".format(i_batch))
            if i_batch <= VAL_THRESHOLD:
                val_dataloader_batches[mark][i_batch] = batch_processor[mark](
                    mini_batch=sample_batched)

        for i_batch, sample_batched in enumerate(eval_dataloader_batches[mark]):
            if i_batch % 100 == 0:
                logger.warning("process_first_cuda_i_batch {}".format(i_batch))
            if i_batch <= VAL_THRESHOLD:
                eval_dataloader_batches[mark][i_batch] = batch_processor[mark](
                    mini_batch=sample_batched)

    try:
        deviceIDs = GPUtil.getAvailable(order='random',
                                        limit=1,
                                        maxLoad=args.maxLoad,
                                        maxMemory=args.maxMemory,
                                        excludeID=excludeID)
        CUDA = 'cuda:' + str(deviceIDs[0])

    except Exception:
        logger.warning("No available device!")

    for model_id, model_config_dict in enumerate(model_config_dicts):
        nasrec_net = Serializer.deserialize(
            jfactory,
            json.dumps(model_config_dict),
            config.ModelConfig(),
        )

        tmp_model = NASRecNet(nasrec_net, feature_config)
        tmp_model.to(device=CUDA)

        svfolder = os.path.join(args.save_model_path, "results", "final_fit")
        svname = os.path.join(svfolder, filenames[model_id].split("/")[-1][:-5] + ".ckp")
        if not os.path.exists(svfolder):
            os.makedirs(svfolder)
        output = simple_train(tmp_model,
                              train_options,
                              train_dataloader,
                              batch_processor[CUDA],
                              CUDA,
                              val_dataloader,
                              0,
                              None,  # send_end,
                              train_dataloader_batches[CUDA],
                              val_dataloader_batches[CUDA],
                              args.batch_size,
                              eval_dataloader,
                              eval_dataloader_batches[CUDA],
                              save_model_name= svname if args.save_model else None,
                              )

        logger.warning("Outputs of Model {} is: {}".format(filenames[model_id], output))
