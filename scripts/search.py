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
import copy

import numpy as np
import torch

from thrift.protocol import TSimpleJSONProtocol
from thrift.util import Serializer

from config import ttypes as config
from models.nas_modules import NASRecNet
from models.builder import load_model
from nasrec.builder import build_searcher, load_searcher, save_searcher
from nasrec.utils import reward_normalization
from trainers.simple import train as simple_train
from utils.data import prepare_data
from utils.search_utils import get_args, get_trainer_config, get_searcher_config

from torch.multiprocessing import Pipe, Process, set_start_method

from thop import profile

set_start_method('spawn', force=True)

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

import GPUtil

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

jfactory = TSimpleJSONProtocol.TSimpleJSONProtocolFactory()

if __name__ == "__main__":
    # get arguments
    args = get_args()

    logger.warning("All Args: {}".format(args))

    # set seeds
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)
    excludeID = [int(id) for id in args.excludeID.split(",")] if args.excludeID else []
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.9, maxMemory=0.8, excludeID=excludeID)
    CUDA = 'cuda:' + str(deviceIDs[0])

    device = torch.device("cpu")
    if args.use_gpu:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device(CUDA)
        else:
            print("WARNING: CUDA is not available on this machine, proceed with CPU")

    # load warm start emb dict
    if args.warm_start_emb:
        if args.data_set_name == "criteo":
           ckp_name = "warm_start_criteo.ckp"
        elif args.data_set_name == "avazu":
           ckp_name = "warm_start_avazu.ckp"
        elif args.data_set_name == "kdd2012":
           ckp_name = "warm_start_kdd2012.ckp"

        warm_start_filename = os.path.join(args.save_model_path, "models", ckp_name)
        warm_start_model = load_model(warm_start_filename)
        warm_start_emb_dict = warm_start_model.emb_dict

    # get trainer config
    input_summary, args = get_trainer_config(args)

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

    # change feature hashing size
    for i, feature in enumerate(input_summary["feature_options"]["sparse"]["features"]):
        if feature["hash_size"] > args.hash_size:
            input_summary["feature_options"]["sparse"]["features"][i]["hash_size"] = args.hash_size

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

    # construct temporal directory to save models
    if args.resume_file:
        temp_dir = os.path.join(
            args.save_model_path,
            args.searcher_type,
            args.data_set_name,
            args.resume_file,
        )
        rewards = np.load(os.path.join(temp_dir, "rewards.npy"), allow_pickle=True).tolist()
        all_roc_aucs = np.load(os.path.join(temp_dir, "all_roc_aucs.npy"), allow_pickle=True).tolist()
        all_arc_vecs = np.load(os.path.join(temp_dir, "all_arc_vecs.npy"), allow_pickle=True).tolist()
        all_actions = np.load(os.path.join(temp_dir, "all_actions.npy"), allow_pickle=True).tolist()
        all_params = np.load(os.path.join(temp_dir, "all_params.npy"), allow_pickle=True).tolist()
        all_flops = np.load(os.path.join(temp_dir, "all_flops.npy"), allow_pickle=True).tolist()
        finished_model = np.load(os.path.join(temp_dir, "finished_model.npy"), allow_pickle=True).tolist()
        fbl_meta = np.load(os.path.join(temp_dir, "fbl_meta.npy"), allow_pickle=True).tolist()
        # unpickling meta data
        with open(os.path.join(temp_dir, "meta.txt"), "rb") as fp:
            [
                best_val_loss,
                best_model,
                best_name,
                best_fbl_id,
                total_model,
                epoch,
            ] = pickle.load(fp)
        fp.close()
        searcher = load_searcher(os.path.join(temp_dir, "searcher.ckp"))

        if args.searcher_type == "evo":
            is_initial = np.load(os.path.join(temp_dir, "is_initial.npy"), allow_pickle=True).tolist()

        if args.searcher_type == "evo":
            searcher.all_arc_vecs = all_arc_vecs
            searcher.all_actions = all_actions
            searcher.all_params = all_params
            searcher.all_flops = all_flops
            searcher.all_rewards = rewards
            searcher.all_roc_aucs = all_roc_aucs
            if args.survival_type == "age":
                searcher.population_arc_queue = all_actions[-searcher.population_size:]
                searcher.population_val_queue = rewards[-searcher.population_size:]
            elif args.survival_type == "comb":
                searcher.comb()
            else:
                if args.survival_type == "fit":
                    idx = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[
                          -searcher.population_size:]
                elif args.survival_type == "mix":
                    division = int(0.5 * searcher.population_size)
                    tmp_rewards = rewards[:-division]
                    idx = sorted(range(len(tmp_rewards)), key=lambda i: tmp_rewards[i], reverse=True)[-division:]
                searcher.population_arc_queue = np.array(all_actions)[idx].tolist()
                searcher.population_val_queue = np.array(rewards)[idx].tolist()
                if args.survival_type == "mix":
                    searcher.population_arc_queue += all_actions[-division:]
                    searcher.population_val_queue += rewards[-division:]
            logger.warning("Total_resume_length: arc_{}, val_{}".format(
                len(searcher.population_arc_queue),
                len(searcher.population_val_queue)
            ))
            searcher.sampler_type = args.sampler_type
            searcher.update_GBDT()
    else:
        if args.save_model_path:
            temp_dir = os.path.join(
                args.save_model_path,
                args.searcher_type,
                args.data_set_name,
                time.strftime("%Y%m%d-%H%M%S"),
            )
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

        # construct searcher
        searcher_config = get_searcher_config(args)
        searcher = build_searcher(searcher_config, feature_config)
        searcher.to(device=device)

        best_val_loss = np.Inf
        best_model = None
        best_name = None
        best_fbl_id = None
        fbl_meta = []
        rewards = []
        all_roc_aucs = []
        finished_model = []
        total_model = -1
        epoch = 0

        # for checking repreated architectures
        all_arc_vecs = []
        # mark all actions (block_configs)
        all_actions = []
        all_params = []
        all_flops = []

        if args.searcher_type == "evo":
            is_initial = True
            all_forward_node_ids = []
            all_virtual_losses = []

    logger.warning("The running history is save in {}".format(temp_dir))

    fbl_run_queue = []
    fbl_result_queue = []
    fbl_device_queue = []
    fbl_time_queue = []
    fbl_name_queue = []
    fbl_id_queue = []
    nasrec_net_queue = []
    nasrec_arc_vec_queue = []
    action_queue = []
    params_queue = []
    flops_queue = []

    # for datasaving purpose
    batch_processor, train_dataloader, val_dataloader, \
    val_dataloader_batches, train_dataloader_batches = {}, {}, {}, {}, {}

    for id in range(args.total_gpus):
        if id not in excludeID:
            CUDA = 'cuda:' + str(id)
            if len(batch_processor) == 0:
                (
                    _,  # datasets
                    batch_processor[CUDA],
                    train_dataloader,
                    val_dataloader,
                    _,  # eval_dataloader
                ) = prepare_data(data_options, performance_options, CUDA)
            else:
                (
                    _,  # datasets
                    batch_processor[CUDA],
                    _,
                    _,
                    _,  # eval_dataloader
                ) = prepare_data(data_options, performance_options, CUDA)
            if args.save_batches:
                train_dataloader_batches[CUDA] = []
                val_dataloader_batches[CUDA] = []
                if len(batch_processor) == 1:
                    for i_batch, sample_batched in enumerate(train_dataloader):
                        if i_batch % 100 == 0:
                            logger.warning("i_batch {}".format(i_batch))
                        train_dataloader_batches[CUDA].append(sample_batched)
                    for i_batch, sample_batched in enumerate(val_dataloader):
                        if i_batch % 100 == 0:
                            logger.warning("i_batch {}".format(i_batch))
                        val_dataloader_batches[CUDA].append(sample_batched)
                    mark = CUDA
                else:
                    train_dataloader_batches[CUDA] = [[]] * len(train_dataloader_batches[mark])
                    val_dataloader_batches[CUDA] = [[]] * len(val_dataloader_batches[mark])
                    for i_batch, sample_batched in enumerate(train_dataloader_batches[mark]):
                        train_dataloader_batches[CUDA][i_batch] = {}
                        if i_batch % 100 == 0:
                            logger.warning("copy i_batch {}".format(i_batch))
                        for k, v in sample_batched.items():
                            train_dataloader_batches[CUDA][i_batch][k] = v.clone().detach()

                    # train_dataloader_batches[CUDA] = train_dataloader_batches[mark]
                    for i_batch, sample_batched in enumerate(train_dataloader_batches[CUDA]):
                        if i_batch % 100 == 0:
                            logger.warning("process_i_batch {}".format(i_batch))
                        train_dataloader_batches[CUDA][i_batch] = batch_processor[CUDA](
                            mini_batch=sample_batched)

                    for i_batch, sample_batched in enumerate(val_dataloader_batches[mark]):
                        val_dataloader_batches[CUDA][i_batch] = {}
                        if i_batch % 100 == 0:
                            logger.warning("copy i_batch {}".format(i_batch))
                        for k, v in sample_batched.items():
                            val_dataloader_batches[CUDA][i_batch][k] = v.clone().detach()

                    # val_dataloader_batches[CUDA] = val_dataloader_batches[mark]
                    for i_batch, sample_batched in enumerate(val_dataloader_batches[CUDA]):
                        if i_batch % 100 == 0:
                            logger.warning("process_i_batch {}".format(i_batch))
                        val_dataloader_batches[CUDA][i_batch] = batch_processor[CUDA](
                            mini_batch=sample_batched)
            else:
                train_dataloader_batches[CUDA] = None
                val_dataloader_batches[CUDA] = None

    if args.save_batches:
        for i_batch, sample_batched in enumerate(train_dataloader_batches[mark]):
            if i_batch % 100 == 0:
                logger.warning("process_first_cuda_i_batch {}".format(i_batch))
            train_dataloader_batches[mark][i_batch] = batch_processor[mark](
                mini_batch=sample_batched)
        for i_batch, sample_batched in enumerate(val_dataloader_batches[mark]):
            if i_batch % 100 == 0:
                logger.warning("process_first_cuda_i_batch {}".format(i_batch))
            val_dataloader_batches[mark][i_batch] = batch_processor[mark](
                mini_batch=sample_batched)

    logger.warning("batch_processor {}".format(batch_processor))


    # load historical samples (could from other searchers)
    if args.historical_sample_path and args.historical_sample_num:
        hist_dir = args.historical_sample_path
        rewards = np.load(os.path.join(hist_dir, "rewards.npy"), allow_pickle=True).tolist()[
                  : args.historical_sample_num
                  ]
        all_actions = np.load(os.path.join(hist_dir, "all_actions.npy"), allow_pickle=True).tolist()[
                      : args.historical_sample_num
                      ]
        # TODO: all_params, all_flops
        try:
            all_params = np.load(os.path.join(hist_dir, "all_params.npy"), allow_pickle=True).tolist()[
                          : args.historical_sample_num
                          ]
            all_flops = np.load(os.path.join(hist_dir, "all_flops.npy"), allow_pickle=True).tolist()[
                          : args.historical_sample_num
                          ]
        except:
            finished_model = np.load(os.path.join(hist_dir, "finished_model.npy"), allow_pickle=True).tolist()
            all_params, all_flops = [], []
            # Get the flops and params of the model
            for i_batch, sample_batched in enumerate(train_dataloader_batches[CUDA]):
                _, feats, _ = sample_batched
                break
            for nasrec_net_fp in finished_model:
                with open(nasrec_net_fp, "r") as fp:
                    nasrec_net_config = json.load(fp)
                nasrec_net = Serializer.deserialize(
                                jfactory,
                                json.dumps(nasrec_net_config),
                                config.ModelConfig(),
                            )
                tmp_model = NASRecNet(nasrec_net, feature_config)

                tmp_model.to(device=CUDA)

                flops, params = profile(tmp_model, inputs=(feats, ), verbose=False)
                flops = flops * 1.0 / args.batch_size
                all_params.append(params)
                all_flops.append(flops)

            np.save(os.path.join(hist_dir, "all_params.npy"), np.array(all_params))
            np.save(os.path.join(hist_dir, "all_flops.npy"), np.array(all_flops))

            all_params = np.load(os.path.join(hist_dir, "all_params.npy"), allow_pickle=True).tolist()[
                          : args.historical_sample_num
                          ]
            all_flops = np.load(os.path.join(hist_dir, "all_flops.npy"), allow_pickle=True).tolist()[
                          : args.historical_sample_num
                          ]
            logger.warning(
                        "resume_all_params: {} all_flops: {}".format(all_params, all_flops)
                    )
        # convert actions to vecs (we do not direcly read the vecs
        # since we may change the vectorized expression of an arc)
        all_arc_vecs = [
            np.concatenate(searcher.dicts_to_vecs(action)) for action in all_actions
        ]

        finished_model = np.load(os.path.join(hist_dir, "finished_model.npy"), allow_pickle=True).tolist()[
                         : args.historical_sample_num
                         ]
        fbl_meta = np.load(os.path.join(hist_dir, "fbl_meta.npy"), allow_pickle=True).tolist()[
                   : args.historical_sample_num
                   ]

        for mp_old in finished_model:
            with open(mp_old, "r") as fp:
                nasrec_net_old = json.load(fp)
            fp.close()
            mp_new = os.path.join(temp_dir, mp_old.split("/")[-1])
            with open(mp_new, "w") as fp:
                json.dump(nasrec_net_old, fp)
            fp.close()

        # unpickling meta data
        best_idx = np.argmin(rewards)
        best_val_loss = rewards[best_idx]
        best_name, best_fbl_id = fbl_meta[best_idx]
        logger.warning(
            "resume_best_val_loss: {} best_idx: {} best_name {}, best_fbl_id {}".format(
                best_val_loss, best_idx, best_name, best_fbl_id
            )
        )
        best_model_filename = os.path.join(
            hist_dir, finished_model[best_idx].split("/")[-1]
        )
        with open(best_model_filename, "r") as fp:
            best_model = json.load(fp)
        fp.close()
        total_model = args.historical_sample_num
        epoch = args.historical_sample_num

        if args.searcher_type == "evo":
            searcher.all_arc_vecs = all_arc_vecs
            searcher.all_actions = all_actions
            searcher.all_params = all_params
            searcher.all_flops = all_flops
            searcher.all_rewards = rewards
            # searcher.all_roc_aucs = all_roc_aucs
            if args.survival_type == "age":
                searcher.population_arc_queue = all_actions[-searcher.population_size:]
                searcher.population_val_queue = rewards[-searcher.population_size:]
            elif args.survival_type == "comb":
                searcher.comb()
            else:
                if args.survival_type == "fit":
                    idx = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[
                          -searcher.population_size:]
                elif args.survival_type == "mix":
                    division = int(0.5 * searcher.population_size)
                    tmp_rewards = rewards[:-division]
                    idx = sorted(range(len(tmp_rewards)), key=lambda i: tmp_rewards[i], reverse=True)[-division:]
                searcher.population_arc_queue = np.array(all_actions)[idx].tolist()
                searcher.population_val_queue = np.array(rewards)[idx].tolist()
                if args.survival_type == "mix":
                    searcher.population_arc_queue += all_actions[-division:]
                    searcher.population_val_queue += rewards[-division:]

            logger.warning("Total_hist_length: arc_{}, val_{}".format(
                len(searcher.population_arc_queue),
                len(searcher.population_val_queue)
            ))

            if len(searcher.population_arc_queue) == searcher.population_size:
                is_initial = False
            searcher.sampler_type = args.sampler_type
            searcher.update_GBDT()

    while epoch < args.search_nepochs:
        while len(fbl_run_queue) < args.num_machines:
            logger.info(
                "Using fblearner training with {} trainers.".format(args.num_trainers)
            )
            # Three steps NAS
            # 1. generate arcs
            if args.searcher_type == "evo":
                nasrec_net, _, actions, nasrec_arc_vecs = searcher.sample(
                    batch_size=1, return_config=True, is_initial=is_initial
                )
            else:
                nasrec_net, log_prob, actions, nasrec_arc_vecs = searcher.sample(
                    batch_size=1, return_config=True
                )
            nasrec_net = nasrec_net[0]
            action = actions[0]
            nasrec_arc_vec = nasrec_arc_vecs[0]
            total_model += 1

            # check if an arch has already been searched before
            repeat_idx = (
                []
                if not all_arc_vecs or args.repeat_checker_off
                else np.where(
                    np.sum(abs(np.array(all_arc_vecs) - nasrec_arc_vec), 1) == 0
                )[0]
            )

            if len(repeat_idx) != 0:
                logger.warning("The architecture is same with: {}.".format(repeat_idx))
                continue

            repeat_idx_1 = (
                []
                if not nasrec_arc_vec_queue or args.repeat_checker_off
                else np.where(
                    np.sum(abs(np.array(nasrec_arc_vec_queue) - nasrec_arc_vec), 1) == 0
                )[0]
            )

            # TODO: check correctness
            if len(repeat_idx_1) != 0:
                logger.warning("The architecture is same with the current running: {}.".format(repeat_idx_1))
                continue

            # 2. put on fblearner to get performance
            model_option = Serializer.serialize(jfactory, nasrec_net)
            model_option = json.loads(model_option)
            input_summary["model_option"] = model_option
            basename = (
                    "[exp autoctr] nasnet_model_search_"
                    + args.searcher_type
                    + "_macro_space_type_"
                    + str(args.macro_space_type)
                    + "_"
                    + str(total_model)
                    + "_updated_model_"
                    + str(epoch)
            )

            try:
                if len(repeat_idx) != 0:
                    break

                # TODO: device
                deviceIDs = GPUtil.getAvailable(order='random',
                                                limit=1,
                                                maxLoad=args.maxLoad,
                                                maxMemory=args.maxMemory,
                                                excludeID=excludeID)
                CUDA = 'cuda:' + str(deviceIDs[0])

            except Exception:
                logger.warning("No available device!")

            try:
                recv_end, send_end = Pipe(False)
                tmp_model = NASRecNet(nasrec_net, feature_config)

                if args.warm_start_emb:
                    tmp_model.emb_dict = copy.deepcopy(warm_start_emb_dict)

                tmp_model.to(device=CUDA)

                # Get the flops and params of the model
                for i_batch, sample_batched in enumerate(train_dataloader_batches[CUDA]):
                    _, feats, _ = sample_batched
                    break
                flops, params = profile(tmp_model, inputs=(feats, ), verbose=False)
                flops = flops * 1.0 / args.batch_size
                logger.warning("The current flops {}, params {}".format(flops, params))

                # launch a subprocess for model training
                new_fbl_run = Process(target=simple_train,
                                      args=(tmp_model,
                                            train_options,
                                            None,  # train_dataloader[CUDA],
                                            batch_processor[CUDA],
                                            CUDA,
                                            None,
                                            0,
                                            send_end,
                                            train_dataloader_batches[CUDA],
                                            val_dataloader_batches[CUDA],
                                            args.batch_size
                                            # args.save_batches,
                                            ))

                new_fbl_run.start()

                fbl_id_queue.append(total_model)
                fbl_run_queue.append(new_fbl_run)
                fbl_result_queue.append(recv_end)
                fbl_device_queue.append(CUDA)
                fbl_time_queue.append(0)
                fbl_name_queue.append(basename)
                nasrec_net_queue.append(model_option)
                nasrec_arc_vec_queue.append(nasrec_arc_vec)
                action_queue.append(action)
                params_queue.append(params)
                flops_queue.append(flops)

            except Exception:
                logger.warning("Model are cannot be registered now!!")

        if len(repeat_idx) != 0:
            # has repeated arch
            (fbl_name, fbl_id) = fbl_meta[repeat_idx[0]]
            rewards.append(rewards[repeat_idx[0]])

            model_filename = finished_model[repeat_idx[0]]

            with open(model_filename, "r") as fp:
                nasrec_net = json.load(fp)
            fp.close()

            nasrec_arc_vec = all_arc_vecs[repeat_idx[0]]
            action = all_actions[repeat_idx[0]]
            params = all_params[repeat_idx[0]]
            flops = all_flops[repeat_idx[0]]

        else:
            # check the status of all the current models
            mark = args.num_machines
            while mark == args.num_machines:
                fbl_time_queue = [t + args.waiting_time for t in fbl_time_queue]
                mark = 0
                for i, fbl_run in enumerate(fbl_run_queue):
                    if (
                            fbl_run.exitcode is None
                            and fbl_time_queue[i] <= args.fbl_kill_time
                    ):
                        mark += 1
                    else:
                        break
                logger.warning("All model are currently running!")
                time.sleep(args.waiting_time)

            # get the terminated workflow
            fbl_run = fbl_run_queue.pop(mark)
            fbl_result = fbl_result_queue.pop(mark)
            fbl_device = fbl_device_queue.pop(mark)
            fbl_time = fbl_time_queue.pop(mark)
            fbl_name = fbl_name_queue.pop(mark).split("_")
            fbl_id = fbl_id_queue.pop(mark)
            nasrec_net = nasrec_net_queue.pop(mark)
            nasrec_arc_vec = nasrec_arc_vec_queue.pop(mark)
            action = action_queue.pop(mark)
            params = params_queue.pop(mark)
            flops = flops_queue.pop(mark)

            if fbl_time > args.fbl_kill_time:
                fbl_run.terminate()
                logger.warning(
                    "Model #_{} training Failed. ID: {}".format(fbl_name[-4], fbl_id)
                )
                epoch -= 1
                continue

            # there exist a model successed in queue
            logger.warning("mark {}, len(fbl_run_queue) {}".format(mark, len(fbl_run_queue)))
            try:
                output = fbl_result.recv()
            except Exception:
                # Failed to extract results due to some transient issue.
                logger.warning(
                    "The results of model #_{} are failed to be obtained. ID: {}. DeviceID: {}".format(
                        fbl_name[-4], fbl_id, fbl_device
                    )
                )
                epoch -= 1
                continue

            logger.warning(
                "Outputs of Model f{}_M_{}_S_{}: {}".format(
                    fbl_id, fbl_name[-4], fbl_name[-1], output
                )
            )

            if output[-2]["avg_val_loss"] is None or np.isnan(output[-2]["avg_val_loss"]) \
                    or output[-2]["roc_auc_score"] is None or np.isnan(output[-2]["roc_auc_score"]):
                # Output is NaN sometimes.
                logger.warning(
                    "Model #_{} validation output is Invalid (None)! ID: {}".format(
                        fbl_name[-4], fbl_id
                    )
                )
                epoch -= 1
                continue

            all_roc_aucs.append([output[-2]["avg_val_loss"], output[-2]["roc_auc_score"]])
            if args.reward_type == "logloss":
                rewards.append(output[-2]["avg_val_loss"])
            elif args.reward_type == "auc":
                rewards.append(1 - output[-2]["roc_auc_score"])

            model_filename = os.path.join(
                temp_dir, "M_" + str(fbl_name[-4]) + "_S_" + str(fbl_name[-1]) + ".json"
            )
        finished_model.append(model_filename)
        fbl_meta.append((fbl_name, fbl_id))
        all_arc_vecs.append(nasrec_arc_vec)
        all_actions.append(action)
        all_params.append(params)
        all_flops.append(flops)
        if args.save_model_path:
            try:
                logger.info("Saving model to {}".format(temp_dir))
                with open(model_filename, "w") as fp:
                    json.dump(nasrec_net, fp)
                fp.close()
                np.save(os.path.join(temp_dir, "rewards.npy"), np.array(rewards))
                np.save(os.path.join(temp_dir, "all_roc_aucs.npy"), np.array(all_roc_aucs))
                np.save(
                    os.path.join(temp_dir, "all_arc_vecs.npy"), np.array(all_arc_vecs)
                )
                np.save(
                    os.path.join(temp_dir, "all_actions.npy"), np.array(all_actions)
                )
                np.save(
                    os.path.join(temp_dir, "all_params.npy"), np.array(all_params)
                )
                np.save(
                    os.path.join(temp_dir, "all_flops.npy"), np.array(all_flops)
                )
                np.save(
                    os.path.join(temp_dir, "finished_model.npy"),
                    np.array(finished_model),
                )
                np.save(os.path.join(temp_dir, "fbl_meta.npy"), np.array(fbl_meta))

                if args.searcher_type == "evo":
                    np.save(os.path.join(temp_dir, "is_initial.npy"), np.array(is_initial))
            except Exception:
                logger.warning("Failed to save the model")

        # update best arc
        if rewards[-1] < best_val_loss:
            best_fbl_id, best_model, best_val_loss, best_name = (
                fbl_id,
                nasrec_net,
                rewards[-1],
                fbl_name,
            )
            if args.save_model_path:
                try:
                    logger.warning("Saving the best model to {}".format(temp_dir))
                    model_filename = os.path.join(temp_dir, "Best_Model" + ".json")
                    with open(model_filename, "w") as fp:
                        json.dump(best_model, fp)
                    fp.close()
                    with open(os.path.join(temp_dir, "best_model_id.txt"), "w") as fp:
                        fp.write(
                            "M_"
                            + str(fbl_name[-4])
                            + "_S_"
                            + str(fbl_name[-1])
                            + ".json"
                            + "\n"
                        )
                    fp.close()
                except Exception:
                    logger.warning("Failed to save the best model")

        # pickling meta data for resume purpose
        if args.save_model_path:
            with open(os.path.join(temp_dir, "meta.txt"), "wb") as fp:
                pickle.dump(
                    [
                        best_val_loss,
                        best_model,
                        best_name,
                        best_fbl_id,
                        total_model,
                        epoch,
                    ],
                    fp,
                )
            fp.close()

        logger.warning(
            "{} model has been finished. The current best arc is: f{}_M_{}_S_{}. Its avg_val_loss is {}.".format(
                len(rewards), best_fbl_id, best_name[-4], best_name[-1], best_val_loss
            )
        )

        # 3. update searcher
        epoch = len(rewards)
        # epoch += 1
        logger.warning("Searcher update epoch {}.".format(epoch))
        if args.searcher_type == "evo":
            searcher.all_arc_vecs = all_arc_vecs
            searcher.all_actions = all_actions
            searcher.all_params = all_params
            searcher.all_flops = all_params
            searcher.all_rewards = rewards
            searcher.all_roc_aucs = all_roc_aucs
            searcher.update([action], [rewards[-1]], survival_type=args.survival_type)

            logger.warning("Total_length update: arc_{}, val_{}".format(
                len(searcher.population_arc_queue),
                len(searcher.population_val_queue)
            ))

            if (
                    is_initial
                    and len(searcher.population_arc_queue) == args.population_size
            ):
                is_initial = False

                for proc in fbl_run_queue:
                    proc.terminate()

                fbl_run_queue = []
                fbl_time_queue = []
                fbl_name_queue = []
                fbl_id_queue = []
                nasrec_net_queue = []
                nasrec_arc_vec_queue = []
                action_queue = []
                params_queue = []
                flops_queue = []

        # save searcher
        save_searcher(os.path.join(temp_dir, "searcher.ckp"), searcher)

    # Kill all remaining workflows on fblearner
    for proc in fbl_run_queue:
        proc.terminate()

    logger.warning(
        "The best arc is: f{}_M_{}_S_{}. Its avg_val_loss is {}.".format(
            best_fbl_id, best_name[-4], best_name[-1], best_val_loss
        )
    )
    logger.warning("\nAll avg_val_loss are {}.".format(rewards))

    if args.save_model_path:
        try:
            logger.warning("Saving the best model to {}".format(temp_dir))
            model_filename = os.path.join(
                temp_dir,
                "Best_Model_M_"
                + str(fbl_name[-4])
                + "_S_"
                + str(fbl_name[-1])
                + ".json",
            )
            with open(model_filename, "w") as fp:
                json.dump(best_model, fp)
            fp.close()
        except Exception:
            logger.warning("Failed to save the best model")
