# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.append('gen-py')

import logging
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ttypes as config


ReaderOption = namedtuple("ReaderOption", ["type", "options"])

logger = logging.getLogger(__name__)

kEpsilon = 1e-10


class DenseDataset(Dataset):
    """Dense dataset."""

    def __init__(self, X, y, sample_weights=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        if sample_weights is not None:
            self.sample_weights = torch.FloatTensor(sample_weights)
        else:
            self.sample_weights = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = self.y[idx]
        sample["dense"] = self.X[idx]
        if self.sample_weights is not None:
            sample["weight"] = self.sample_weights[idx]
        return sample

    def share_memory_(self):
        self.X.share_memory_()
        self.y.share_memory_()
        if self.sample_weights is not None:
            self.sample_weights.share_memory_()


############################################################
# criteo data utils
############################################################


class CriteoDataset(Dataset):
    """Criteo dataset."""

    def __init__(self, X_cat, X_int, y, dense_transform=None):
        self.X_cat, self.X_int, self.y, self.dense_transform = (
            torch.LongTensor(X_cat),
            torch.FloatTensor(X_int),
            torch.FloatTensor(y),
            dense_transform,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #  Criteo data only have categorical features as sparse feature
        sample = {
            "sparse_{}".format(i): torch.tensor([v + 1])
            for i, v in enumerate(self.X_cat[idx])
        }
        sample["label"] = self.y[idx]
        sample["dense"] = (
            self.X_int[idx]
            if self.dense_transform is None
            else self.dense_transform(self.X_int[idx])
        )
        return sample

    def share_memory_(self):
        self.X_cat.share_memory_()
        self.X_int.share_memory_()
        self.y.share_memory_()


############################################################
# synthetic data utils
############################################################


def _set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(
        self,
        num_dense,
        num_sparse,
        max_sparse_id,
        num_samples,
        batch_size,
        id_list_configs,
    ):
        _set_random_seed()

        # We generate 10k examples and then reuse the these examples during
        # data reading
        data_num_samples = 10000
        self.num_batches = data_num_samples // batch_size
        self.num_batch_samples = num_samples // batch_size

        # Limit the number of examples as we are only doing benchmarking for
        # synthetic data set.
        dense = torch.randn((self.num_batches, batch_size, num_dense))
        label = torch.randint(2, size=(self.num_batches, batch_size))
        weight = None  # torch.ones((self.num_batches, batch_size))
        assert (
            len(id_list_configs) == num_sparse or len(id_list_configs) == 1
        ), "len(id_list_configs) != num_sparse: {0} vs {1}".format(
            len(id_list_configs), num_sparse
        )
        if len(id_list_configs) == 1:
            id_list_configs = [deepcopy(id_list_configs[0]) for _ in range(num_sparse)]

        sparse_id_list_len = [
            [
                [
                    min(max(0, int(x)), config.truncation)
                    for x in np.random.normal(
                        config.mean, config.std, size=(batch_size)
                    )
                ]
                for config in id_list_configs
            ]
            for _ in range(self.num_batches)
        ]

        sparse = []
        for k in range(self.num_batches):
            sparse_batch = []
            for i in range(num_sparse):
                sparse_batch.append({})
                ids = []
                offsets = [0]
                for j in range(batch_size):
                    id_list_len = sparse_id_list_len[k][i][j]
                    ids.extend(np.random.randint(max_sparse_id, size=id_list_len))
                    offsets.append(offsets[-1] + id_list_len)
                sparse_batch[i]["data"] = torch.tensor(ids)
                sparse_batch[i]["offsets"] = torch.tensor(offsets[:-1])
            sparse.append(sparse_batch)

        self.data = []
        for i in range(self.num_batches):
            batch = {}
            batch["dense"] = dense[i]
            batch["label"] = label[i]
            batch["weight"] = weight[i] if weight is not None else None
            batch["sparse"] = [sparse[i][j] for j in range(num_sparse)]
            self.data.append(batch)

    def __len__(self):
        return self.num_batch_samples

    def __getitem__(self, idx):
        return self.data[idx % self.num_batches]


def synthetic_data_generator(
    num_dense, num_sparse, max_sparse_id, num_samples, batch_size, id_list_configs
):
    _set_random_seed()
    # Limit the number of examples as we are only doing benchmarking for
    # synthetic data set.
    data_num_batches = min(1000, min(100000, num_samples) // batch_size)
    dense = torch.randn((data_num_batches, batch_size, num_dense))
    label = torch.randint(2, size=(data_num_batches, batch_size))
    # weight = torch.ones((data_num_batches, batch_size))
    assert (
        len(id_list_configs) == num_sparse or len(id_list_configs) == 1
    ), "len(id_list_configs) != num_sparse: {0} vs {1}".format(
        len(id_list_configs), num_sparse
    )
    if len(id_list_configs) == 1:
        id_list_configs = [deepcopy(id_list_configs[0]) for _ in range(num_sparse)]

    sparse_id_list_len = [
        [
            [
                min(max(0, int(x)), config.truncation)
                for x in np.random.normal(config.mean, config.std, size=(batch_size))
            ]
            for config in id_list_configs
        ]
        for _ in range(data_num_batches)
    ]

    sparse = []
    for k in range(data_num_batches):
        sparse_batch = []
        for i in range(num_sparse):
            sparse_batch.append({})
            ids = []
            offsets = [0]
            for j in range(batch_size):
                id_list_len = sparse_id_list_len[k][i][j]
                ids.extend(np.random.randint(max_sparse_id, size=id_list_len))
                offsets.append(offsets[-1] + id_list_len)
            sparse_batch[i]["data"] = torch.tensor(ids)
            sparse_batch[i]["offsets"] = torch.tensor(offsets[:-1])
        sparse.append(sparse_batch)

    data = []
    for i in range(data_num_batches):
        batch = {}
        batch["dense"] = dense[i]
        batch["label"] = label[i]
        batch["weight"] = None  # weight[i]
        batch["sparse"] = [sparse[i][j] for j in range(num_sparse)]
        data.append(batch)
    return data


def get_split_indices(splits, num_samples):
    if np.sum(splits) >= 1.0:
        raise ValueError("sum of splits should be smaller than 1.0")

    bins = list(np.cumsum([0.0] + list(splits)))
    bins.append(1.0)
    indices = [
        range(int(bins[i] * num_samples), int(bins[i + 1] * num_samples))
        for i in range(len(splits) + 1)
    ]
    if any(len(indice) <= 0 for indice in indices):
        raise ValueError(
            "Split {} is causing empty partitions: {}".format(
                splits, [len(indice) for indice in indices]
            )
        )
    return indices


def split_dense_dataset(data, splits, sample_weights=None):
    """
    dataset: Dataset
    splits: array of split ratio of length L, will create L+1 dataloaders
            according to the ratio, the last partition is 1.0-sum(splits);
            if None, return the entire dataset in dataloader
            example:
                splits= [0.8, 0.1] for a 80%, 10%, 10% splits
                between train, validation, eval
    """

    num_samples = len(data["y"])
    indices = get_split_indices(splits=splits, num_samples=num_samples)
    logger.info(
        "Split data into partitions with size: {}".format(
            [len(indice) for indice in indices]
        )
    )

    datasets = []
    for indice in indices:
        dataset = DenseDataset(
            data["X"][indice],
            data["y"][indice],
            None if sample_weights is None else sample_weights[indice],
        )
        datasets.append(dataset)
    return datasets


def load_and_split_dataset(npz_file, splits=None):
    """
    dataset: Dataset
    splits: array of split ratio of length L, will create L+1 dataloaders
            according to the ratio, the last partition is 1.0-sum(splits);
            if None, return the entire dataset in dataloader
            example:
                splits= [0.8, 0.1] for a 80%, 10%, 10% splits
                between train, validation, eval
    """
    data = np.load(npz_file)

    if splits is None:
        return CriteoDataset(X_cat=data["X_cat"], X_int=data["X_int"], y=data["y"])

    num_samples = len(data["y"])
    indices = get_split_indices(splits=splits, num_samples=num_samples)
    logger.info(
        "Split data into partitions with size: {}".format(
            [len(indice) for indice in indices]
        )
    )
    return [
        CriteoDataset(
            X_cat=data["X_cat"][indice],
            X_int=data["X_int"][indice],
            y=data["y"][indice],
        )
        for indice in indices
    ]


############################################################
# batch processors
############################################################


def _save_transforms(dense_transform, filename):
    torch.save({"dense_transform": dense_transform}, filename)


def _load_transforms(filename):
    state = torch.load(filename)
    return state["dense_transform"]


# the __call__ method for a BatchProcessor should return label, feats, weight:
# label: a (batch_size,) FloatTensor for labels
# weight: optional, None or (batch_size,) FloatTensor for per sample weights
# feats: dict for features
#     feats['dense']:  (batch_size, num_dense) FloatTensor for dense features
#     feats['[sparse_feature_name]]']: for each sparse feature name (consistent
#           with feature_config), it is a dict with two keys:
#           'data' and 'offsets'. See EmbeddingBag doc for the supported types.
class BatchProcessor(object):
    def __init__(
        self,
        feature_config=None,
        dense_transform=None,
        device=None,
        dense_feature_clamp=-1.0,
    ):
        self.feature_config = deepcopy(feature_config)
        self.dense_transform = dense_transform
        self.device = torch.device("cpu") if device is None else device
        self.dense_feature_clamp = dense_feature_clamp

    def save_transforms(self, filename):
        _save_transforms(self.dense_transform, filename)

    def load_transforms(self, filename):
        self.dense_transform = _load_transforms(filename)

    def share_memory(self):
        if self.dense_transform is not None:
            self.dense_transform.share_memory_()

    def __call__(self):
        raise NotImplementedError


class DenseBatchProcessor(BatchProcessor):
    def __call__(self, mini_batch):
        for k, v in mini_batch.items():
            if k == "dense":
                v = v if self.dense_transform is None else self.dense_transform(v)
                mini_batch[k] = v.to(device=self.device, dtype=torch.float32)
            elif k in ["label", "weight"]:
                mini_batch[k] = v.to(device=self.device, dtype=torch.float32)
            else:
                raise ValueError("invalid mini_batch key")

        label = mini_batch.pop("label", None)
        weight = mini_batch.pop("weight", None)
        return label, mini_batch, weight


class CriteoBatchProcessor(BatchProcessor):
    def __call__(self, mini_batch, transform=True, reverse=0):
        if reverse == 1:
            for k, v in mini_batch.items():
                if k in ["dense", "label"]:
                    mini_batch[k] = v.to(device=self.device, dtype=torch.float32)
                else:
                    mini_batch[k] = {
                        "data": v["data"].to(device=self.device, dtype=torch.long),
                        "offsets": None,
                    }
        elif reverse == 2:
            for k, v in mini_batch.items():
                if k in ["dense", "label"]:
                    mini_batch[k] = v.to(device=torch.device("cpu"), dtype=torch.float32)
                else:
                    mini_batch[k] = {
                        "data": v["data"].to(device=torch.device("cpu"), dtype=torch.long),
                        "offsets": None,
                    }
        else:
            if transform:
                for k, v in mini_batch.items():
                    if k == "dense":
                        v = v if self.dense_transform is None else self.dense_transform(v)
                        mini_batch[k] = v.to(device=self.device, dtype=torch.float32)
                    elif k == "label":
                        mini_batch[k] = v.to(device=self.device, dtype=torch.float32)
                    else:
                        mini_batch[k] = {
                            "data": v.to(device=self.device, dtype=torch.long),
                            "offsets": None,
                        }
            # else:
                # for k, v in mini_batch.items():
                #     mini_batch[k] = v


            # label = mini_batch.pop("label", None)
        label = mini_batch["label"]
        # Criteo does not have sample weights
        weight = None
        return label, mini_batch, weight


def loadDataset(file):
    """
    Loads dataset from NumPy format.

    Inputs:
        file (str): path to the npz file of dataset (Kaggle or Terabyte)

    Outputs:
        X_cat (np.ndarray): categorical features
        X_int (np.ndarray): continuous features
        y (np.ndarray): labels
        counts (list): number of categories for each categorical feature

    """
    # load and preprocess data
    with np.load(file) as data:
        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    return X_cat, X_int, y, counts


############################################################
# dense transform
############################################################


class DenseTransform(object):
    def __init__(self, mean, std):
        self.mean = mean.cpu()
        self.std = std.cpu()

    def __call__(self, dense):
        return (dense - self.mean) / self.std

    def share_memory_(self):
        self.mean.share_memory_()
        self.std.share_memory_()


def create_dense_transform(train_dataloader, batch_processor, num_batches):
    mean = 0.0
    num_samples = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        if i_batch >= num_batches:
            break
        _, feats, _ = batch_processor(mini_batch=sample_batched)
        dense = feats["dense"]
        num_samples += dense.shape[0]
        mean += torch.sum(dense.to(dtype=torch.float), dim=0)
    mean /= num_samples

    var = 0.0
    num_samples = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        if i_batch >= num_batches:
            break
        _, feats, _ = batch_processor(mini_batch=sample_batched)
        dense = feats["dense"]
        num_samples += dense.shape[0]
        var += torch.sum((dense.to(dtype=torch.float) - mean) ** 2, dim=0)
    std = torch.sqrt((var + kEpsilon) / num_samples)
    return DenseTransform(mean=mean, std=std)


def create_dense_transform_from_synthetic():
    # Due to the dense features are sampled from normal distribution,
    # we simply set mean and std based on normal distribution.
    # We add this part is for benchmark purpose.
    return DenseTransform(mean=torch.tensor(0), std=torch.tensor(1))


def prepare_data(data_options, performance_options, CUDA="cuda:0", pin_memory=False):
    if data_options.getType() == config.DataConfig.FROM_FILE:
        data_option = data_options.get_from_file()
        (
            datasets,
            batch_processor,
            train_dataloader,
            val_dataloader,
            eval_dataloader,
        ) = prepare_criteo_data(data_option, performance_options, CUDA, pin_memory)
    else:
        raise ValueError("Unknown data option type.")
    dense_transform = create_dense_transform(
        train_dataloader,
        batch_processor,
        num_batches=int(data_option.num_samples_meta / data_option.batch_size),
    )
    batch_processor.dense_transform = dense_transform
    return datasets, batch_processor, train_dataloader, val_dataloader, eval_dataloader


def prepare_criteo_data(data_options, performance_options, CUDA, pin_memory=False):
    logger.info("Loading data from {}".format(data_options.data_file))
    datasets = load_and_split_dataset(
        npz_file=data_options.data_file, splits=data_options.splits
    )
    logger.info("Data loaded")
    # pin_memory=True,
    train_dataloader, val_dataloader, eval_dataloader = (
        DataLoader(dataset,
                    batch_size=data_options.batch_size,
                    pin_memory=pin_memory,
                    num_workers=performance_options.num_readers) for dataset in datasets
    )
    batch_processor = CriteoBatchProcessor(
        device=(
            torch.device(CUDA)
            if performance_options.use_gpu
            else torch.device("cpu")
        )
    )
    return datasets, batch_processor, train_dataloader, val_dataloader, eval_dataloader
