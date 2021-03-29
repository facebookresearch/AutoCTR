# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re
import time

import numpy as np
import torch


logger = logging.getLogger(__name__)


def log_train_info(
    start_time,
    i_batch="N/A",
    i_epoch="N/A",
    trainer_id="N/A",
    num_batches=1,
    total_loss=0,
    batch_size=None,
    num_samples=None,
    sample_weight_sum=None,
    ctr=None,
    lock=None,
    on_gpu=False,
    trainer_logger=None,
):
    """
    Args:
        total_loss, the sum of the averaged per batch loss
    """
    if on_gpu:
        torch.cuda.synchronize()

    curr_time = time.time()

    if trainer_logger is None:
        trainer_logger = logger

    if lock is not None:
        lock.acquire()

    try:
        if num_samples is None:
            assert (
                batch_size is not None
            ), "batch_size and num_samples cannot both be None."
            num_samples = num_batches * batch_size
        if sample_weight_sum is None:
            assert (
                batch_size is not None
            ), "batch_size and sample_weight_sum cannot both be None."
            sample_weight_sum = num_batches * batch_size
        loss = total_loss / sample_weight_sum
        ne = calculate_ne(loss, ctr) if ctr is not None else "N/A"
        trainer_logger.warning(
            "Trainer {} finished iteration {} of epoch {}, "
            "{:.2f} qps, "
            "window loss: {}, "
            "window NE: {}".format(
                trainer_id,
                i_batch,
                i_epoch,
                num_samples / (curr_time - start_time),
                loss,
                ne,
            )
        )
    finally:
        if lock is not None:
            lock.release()

    return (loss, ne) if ctr is not None else loss


log_eval_info = log_train_info


def log_tb_info_batch(
    writer,
    model,
    pred,
    label,
    optimizer,  # not used
    logging_options,
    iter,
    start_time,
    trainer_id=None,
    total_loss=0,
    batch_size=-1,  # not used
    num_batches=-1,  # not used
    sample_weight_sum=None,
    avg_loss=None,
    ctr=None,
    lock=None,
):
    """
    Note that the reported value is the mean of per batch mean,
    which is different from mean of the whole history

    Args:
        total_loss, the sum of the averaged per batch loss
    """
    if writer is None:
        return
    if lock is not None:
        lock.acquire()
    try:
        if avg_loss is None:
            assert (
                total_loss is not None and sample_weight_sum is not None
            ), "cannot compute avg_loss"
            avg_loss = total_loss / sample_weight_sum
        writer.add_scalar(
            "{}batch/train_metric/loss".format(
                "" if trainer_id is None else "trainer_{}/".format(trainer_id)
            ),
            avg_loss,
            iter,
        )
        if ctr is not None:
            ne = calculate_ne(avg_loss, ctr)
            writer.add_scalar(
                "{}batch/train_metric/ne".format(
                    "" if trainer_id is None else "trainer_{}/".format(trainer_id)
                ),
                ne,
                iter,
            )
        if logging_options.tb_log_pr_curve_batch:
            writer.add_pr_curve("PR Curve", label, pred, iter)
        if logging_options.tb_log_model_weight_hist:
            for name, param in model.named_parameters():
                if any(
                    re.search(pattern, name)
                    for pattern in logging_options.tb_log_model_weight_filter_regex
                ):
                    continue
                writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)
    finally:
        if lock is not None:
            lock.release()


def need_to_log_batch(counter, logging_options, batch_size):
    return (
        logging_options.log_freq > 0
        and (counter + 1) % max(1, int(logging_options.log_freq / batch_size)) == 0
    )


def need_to_log_tb(counter, logging_options, batch_size):
    tb_log_freq = logging_options.tb_log_freq
    return (
        tb_log_freq > 0 and (counter + 1) % max(1, int(tb_log_freq / batch_size)) == 0
    )


def is_checkpoint(counter, ckp_interval, ckp_path):
    return ckp_interval > 0 and ckp_path and (counter + 1) % ckp_interval == 0


def calculate_ne(logloss, ctr):
    if ctr <= 0.0 or ctr >= 1.0:
        logger.error("CTR should be between 0.0 and 1.0")
        return 0.0 if logloss == 0.0 else np.inf
    return -logloss / (ctr * np.log(ctr) + (1.0 - ctr) * np.log(1 - ctr))
