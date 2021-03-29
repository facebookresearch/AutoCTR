# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import math
import numpy as np
import lightgbm as lgb
import scipy.stats as ss

from config import ttypes as config
from models.nas_modules import NASRecNet
from .base_searcher import BaseSearcher

logger = logging.getLogger(__name__)


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def prob_comb(population_size, candidate_size):
    prob = []
    for rank in range(population_size, 0, -1):
        prob.append(nCr(rank + candidate_size-1, candidate_size)/nCr(population_size + candidate_size, candidate_size + 1))
    return prob

class EvolutionaryController(BaseSearcher):
    """Aging evolution: https://arxiv.org/abs/1802.01548
    """

    def __init__(self, searcher_config, feature_config):
        super(EvolutionaryController, self).__init__(searcher_config, feature_config)
        self.controller_option = searcher_config.get_evolutionary_searcher()
        self._init_base_searcher_params()
        self.population_size = self.controller_option.population_size
        self.candidate_size = self.controller_option.candidate_size
        self.all_arc_vecs = None
        self.all_rewards = None
        self.all_params = None
        self.all_flops = None
        self.sampler_type = 1
        self.eval_at = 3
        self._build_arc()
        self.sample_prob = prob_comb(self.population_size, self.candidate_size)

    def _build_arc(self):
        self.population_arc_queue = []
        self.population_val_queue = []

    def _selection_candidate(self, type=0):
        if type == 0:
            candidate_indices = np.sort(
                np.random.choice(
                    self.population_size, self.candidate_size, replace=False
                )
            )
            candidate_arcs = list(
                map(self.population_arc_queue.__getitem__, candidate_indices)
            )
            candidate_vals = list(
                map(self.population_val_queue.__getitem__, candidate_indices)
            )
            best_arc_idx = np.argmin(candidate_vals)
            best_arc = candidate_arcs[best_arc_idx]
        elif type == 1:

            rank = ss.rankdata(np.array(self.population_val_queue), method='ordinal')
            tmp_prob = [self.sample_prob[i-1] for i in rank]
            best_arc_idx = np.random.choice(list(range(self.population_size)), p=tmp_prob)
            best_arc = self.population_arc_queue[best_arc_idx]

        return best_arc_idx, best_arc

    def sample(self, batch_size=1, return_config=False, is_initial=True):
        """sample a batch_size number of NasRecNets from the controller, where
        each node is made up of a set of blocks with number self.num_blocks.
        If is_initial=True, random sample a batch size of arcs into population,
        else sample a candidate size arch from population queue, get the best one,
        mutate the best one to a new arch, repeat this a batch_size of time.
        """
        if batch_size < 1:
            raise ValueError("Wrong batch_size.")

        nasrec_nets, all_vec_configs, nasrec_arc_vecs = [], [], []
        for _ in range(batch_size):
            if is_initial:
                vecs, vec_configs = self.random_sample()
            else:
                best_arc_idx, best_arc = self._selection_candidate(type=1)

                # mutate to get child
                if self.sampler_type > 1:
                    vecs, vec_configs = self.ML_sampler(parent=best_arc)
                else:
                    vecs, vec_configs = self.mutate_arc(parent=best_arc)

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

    def update(self, actions, rewards, survival_type="age"):
        """add k new archs into the population queue and
        kick out the k oldest archs"""

        # add child to right of population
        self.population_arc_queue += actions
        self.population_val_queue += rewards

        if survival_type == "age":
            self.population_arc_queue = self.population_arc_queue[-self.population_size:]
            self.population_val_queue = self.population_val_queue[-self.population_size:]
        elif survival_type == "comb":
            self.comb()
        else:
            if survival_type == "fit":
                idx = sorted(
                    range(len(self.population_val_queue)),
                    key=lambda i: self.population_val_queue[i], reverse=True
                )[-self.population_size:]
            elif survival_type == "mix":
                division = int(0.5 * self.population_size)
                tmp_rewards = self.population_val_queue[:-division]
                idx = sorted(range(len(tmp_rewards)), key=lambda i: tmp_rewards[i], reverse=True)[-division:]
                age_arcs = self.population_arc_queue[-division:]
                age_vals = self.population_val_queue[-division:]
            self.population_arc_queue = np.array(self.population_arc_queue)[idx].tolist()
            self.population_val_queue = np.array(self.population_val_queue)[idx].tolist()
            if survival_type == "mix":
                self.population_arc_queue += age_arcs
                self.population_val_queue += age_vals

        # if keep_largest:
        #     idx = sorted(
        #         range(len(self.population_val_queue)),
        #         key=lambda i: self.population_val_queue[i], reverse=True
        #     )[-self.population_size:]
        #     self.population_arc_queue = np.array(self.population_arc_queue)[idx].tolist()
        #     self.population_val_queue = np.array(self.population_val_queue)[idx].tolist()
        # else:
        #     # remove dead from left of population if exceed population_size
        #     self.population_arc_queue = self.population_arc_queue[-self.population_size :]
        #     self.population_val_queue = self.population_val_queue[-self.population_size :]

        if self.sampler_type > 1:
            # QQ TODO: build GBDT_rank:
            self.update_GBDT()

    def comb(self, trade_off=[0.1, 1, 0.1, 1]):

        if len(self.all_rewards) <= self.population_size:
            self.population_arc_queue = self.all_actions[-self.population_size:]
            self.population_val_queue = self.all_rewards[-self.population_size:]
        else:
            if trade_off[3] == 0:
                rank_weight = ss.rankdata(np.array(self.all_rewards)) / len(self.all_rewards)
                age_weight = np.array(range(len(self.all_rewards), 0, -1)) / len(self.all_rewards)
                age_weight[:self.population_size] = age_weight[self.population_size - 1]
                flops_weight = ss.rankdata(np.array(self.all_flops)) / len(self.all_flops)
                all_weight = trade_off[0] * rank_weight + trade_off[1] * age_weight + trade_off[2] * flops_weight
                idx = np.array(
                    sorted(range(len(all_weight)), key=lambda i: all_weight[i]))[:self.population_size]# < self.population_size  # [-division:]
                self.population_arc_queue = np.array(self.all_actions)[idx].tolist()
                self.population_val_queue = np.array(self.all_rewards)[idx].tolist()
            elif trade_off[3] == 1:
                age_weight = np.array(range(len(self.all_rewards), 0, -1)) / len(self.all_rewards)
                age_weight[:self.population_size] = age_weight[self.population_size - 1]
                # filter with age weight
                idx1 = np.array(
                      sorted(range(len(age_weight)), key=lambda i: age_weight[i]))[:2*self.population_size]

                age_rewards = np.array(self.all_rewards)[idx1].tolist()
                age_actions = np.array(self.all_actions)[idx1].tolist()
                age_flops = np.array(self.all_flops)[idx1].tolist()

                rank_weight = ss.rankdata(np.array(age_rewards)) / len(age_rewards)
                age_weight = np.array(age_weight)[idx1]
                flops_weight = ss.rankdata(np.array(age_flops)) / len(age_flops)

                all_weight = trade_off[0] * rank_weight +  trade_off[1] * age_weight + trade_off[2] * flops_weight
                idx2 = np.array(
                    sorted(range(len(all_weight)), key=lambda i: all_weight[i]))[:self.population_size] # < self.population_size  # [-division:]
                self.population_arc_queue = np.array(age_actions)[idx2].tolist()
                self.population_val_queue = np.array(age_rewards)[idx2].tolist()

    def update_GBDT(self):

        k = len(self.all_arc_vecs)
        r = 0.8

        # create dataset for lightgbm
        X_train, X_test, y_train1, y_test1 = self.all_arc_vecs[:int(k * r)], \
                                             self.all_arc_vecs[int(k * r):], \
                                             self.all_rewards[:int(k * r)], \
                                             self.all_rewards[int(k * r):]

        X_train, X_test, y_train1, y_test1 = np.array(X_train), \
                                             np.array(X_test), \
                                             np.array(y_train1), \
                                             np.array(y_test1)
        logger.warning('Train Shape {}{}{}{}'.format(X_train.shape,
                                                     X_test.shape,
                                                     y_train1.shape,
                                                     y_test1.shape))

        y_train = ss.rankdata(-y_train1) - 1
        y_test = ss.rankdata(-y_test1) - 1
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        lgb_train = lgb.Dataset(X_train, y_train, group=np.array([len(y_train)]))  # free_raw_data=False
        lgb_eval = lgb.Dataset(X_test, y_test, group=np.array([len(y_test)]),
                               reference=lgb_train)  # ，free_raw_data=False

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'lambdarank', # 'regression',  #
            'metric': "ndcg",  # "auc", #"ndcg", # {'l2', 'l1'},
            'label_gain': np.array(list(range(len(y_train)))) * 2,  #
            'max_depth': 3,  # 'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'eval_at': self.eval_at,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': 5,
        }

        logger.warning('Starting training...')

        # train
        self.gbm = lgb.train(params,
                             lgb_train,
                             num_boost_round=1500,
                             valid_sets=lgb_eval,
                             early_stopping_rounds=150)
        logger.warning('Finish training...')

    def ML_sampler(self, parent):
        vecs_list, arc_vec_list, vec_configs_list = [], [], []
        i = 0
        while i < self.sampler_type:
            vecs, vec_configs = self.mutate_arc(parent=parent)
            arc_vec = np.concatenate(vecs)

            # check current
            repeat_idx = (
                []
                if not arc_vec_list
                else np.where(
                    np.sum(abs(np.array(arc_vec_list) - arc_vec), 1) == 0
                )[0]
            )

            if len(repeat_idx) != 0:
                logger.warning("The architecture is same with: {}.".format(repeat_idx))
                continue

            # check all
            repeat_idx = (
                []
                if not self.all_arc_vecs
                else np.where(
                    np.sum(abs(np.array(self.all_arc_vecs) - arc_vec), 1) == 0
                )[0]
            )

            if len(repeat_idx) != 0:
                logger.warning("The architecture is same all_arc_vectors with: {}.".format(repeat_idx))
                continue

            vecs_list.append(vecs)
            arc_vec_list.append(arc_vec)
            vec_configs_list.append(vec_configs)
            i += 1

        logger.warning('Test Shape {}'.format(np.array(arc_vec_list).shape))

        y_pred = self.gbm.predict(np.array(arc_vec_list), num_iteration=self.gbm.best_iteration)

        idx = np.where(y_pred == np.max(y_pred))[0][0]

        return vecs_list[idx], vec_configs_list[idx]
