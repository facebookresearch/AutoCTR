# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .loss import apply_loss, build_loss
from .utils import log_tb_info_batch, log_train_info, need_to_log_batch, need_to_log_tb
from models.builder import save_model

try:
    from fblearner.flow.util.visualization_utils import summary_writer
except ImportError:
    pass

logger = logging.getLogger(__name__)

np.set_printoptions(precision=5)
torch.set_printoptions(precision=5)

THRESHOLD = -1 # -1 #7500
VAL_THRESHOLD = -1


def train(
        model,
        train_options,
        train_dataloader=None,
        batch_processor=None,
        device=None,
        val_dataloader=None,
        trainer_id=0,
        send_end=None,
        train_dataloader_batches=None,
        val_dataloader_batches=None,
        batch_size=1024,
        eval_dataloader=None,
        eval_dataloader_batches=None,
        save_model_name=None,
):
    try:
        writer = summary_writer()
    except Exception:
        logger.error("Failed to create the tensorboard summary writer.")
        writer = None

    prev_avg_val_loss, is_improving, is_local_optimal = None, True, False
    optimizer = model.get_optimizers()
    loss = build_loss(model, loss_config=train_options.loss)
    output = []
    logging_options = train_options.logging_config
    batch_size = batch_size

    if train_dataloader_batches is None:
        train_dataloader_batches = train_dataloader
        is_train_dataloader = True
    else:
        is_train_dataloader = False

    if val_dataloader_batches is None:
        val_dataloader_batches = val_dataloader
        is_val_dataloader = True
    else:
        is_val_dataloader = False

    if eval_dataloader_batches is None:
        eval_dataloader_batches = eval_dataloader
        is_eval_dataloader = True
    else:
        is_eval_dataloader = False

    for i_epoch in range(0, train_options.nepochs):
        start_time_epoch = time.time()
        num_batches, avg_loss_epoch, q1, q2 = train_epoch(
            model=model,
            loss=loss,
            optimizer=optimizer,
            batch_processor=batch_processor,
            trainer_id=trainer_id,
            i_epoch=i_epoch,
            device=device,
            logging_options=logging_options,
            writer=writer,
            train_dataloader_batches=train_dataloader_batches,
            batch_size=batch_size,
            is_dataloader=is_train_dataloader,
        )

        logger.warning("Epoch:{}, Time for training: {}".format(i_epoch, time.time() - start_time_epoch))

        avg_loss_epoch = log_train_info(
            start_time=start_time_epoch,
            i_batch=num_batches,
            i_epoch=i_epoch,
            trainer_id=trainer_id,
            total_loss=avg_loss_epoch * num_batches * batch_size,
            num_batches=num_batches,
            batch_size=batch_size,
        )
        if writer is not None:
            writer.add_scalar("train_metric/loss_epoch", avg_loss_epoch, i_epoch)
        output.append({"i_epoch": i_epoch, "avg_train_loss": avg_loss_epoch})

        if val_dataloader_batches is not None:
            avg_val_loss, _, _, avg_auc = evaluate(
                model=model,
                loss=loss,
                dataloader=val_dataloader_batches,
                batch_processor=batch_processor,
                device=device,
                batch_size=batch_size,
                is_dataloader=is_val_dataloader,
                i_epoch=i_epoch,
            )
            output[-1]["avg_val_loss"] = avg_val_loss
            output[-1]["roc_auc_score"] = avg_auc

            if eval_dataloader_batches is not None:
                avg_eval_loss, _, _, avg_eval_auc = evaluate(
                    model=model,
                    loss=loss,
                    dataloader=eval_dataloader_batches,
                    batch_processor=batch_processor,
                    device=device,
                    batch_size=batch_size,
                    is_dataloader=is_eval_dataloader,
                    i_epoch=i_epoch,
                )
                output[-1]["avg_eval_loss"] = avg_eval_loss
                output[-1]["eval_roc_auc_score"] = avg_eval_auc

                # check if local optimal
            (
                is_local_optimal,
                is_improving,
                prev_avg_val_loss,
            ) = _check_local_optimal(
                i_epoch, is_improving, avg_val_loss, prev_avg_val_loss
            )

            # break if is local optimal
            if is_local_optimal and train_options.early_stop_on_val_loss:
                break

            if save_model_name:
                save_model(save_model_name, model)

            if writer is not None:
                writer.add_scalar("val_metric/loss_epoch", avg_val_loss, i_epoch)
            logger.warning("Epoch:{}, validation loss: {}, roc_auc_score: {}, time: {}, q1: {}, q2: {}".format(i_epoch,
                                                                                                               avg_val_loss,
                                                                                                               avg_auc,
                                                                                                               time.time() - start_time_epoch,
                                                                                                               np.sum(
                                                                                                                   q1),
                                                                                                               np.sum(
                                                                                                                   q2)))
    if writer is not None:
        writer.close()
    if send_end:
        send_end.send(output)
    return output


def _check_local_optimal(i_epoch, is_improving, avg_val_loss, prev_avg_val_loss):
    is_local_optimal = i_epoch > 0 and is_improving and avg_val_loss > prev_avg_val_loss
    is_improving = i_epoch == 0 or prev_avg_val_loss > avg_val_loss
    prev_avg_val_loss = avg_val_loss
    return is_local_optimal, is_improving, prev_avg_val_loss


def train_epoch(
        model,
        loss,
        optimizer,
        batch_processor,
        logging_options,
        device,
        trainer_id,
        i_epoch,
        lock=None,
        writer=None,
        train_dataloader_batches=None,
        batch_size=1024,
        is_dataloader=True,
):
    model.train()
    start_time, loss_val, num_batches, sample_weight_sum = time.time(), 0.0, 0, 0.0
    start_time_tb, loss_val_tb, sample_weight_sum_tb = (time.time(), 0.0, 0.0)

    loss_val_epoch, total_num_batches, sample_weight_sum_epoch = (
        0.0,
        len(train_dataloader_batches),
        0.0,
    )
    batch_size = batch_size

    q1, q2 = [], []

    qq3 = time.perf_counter()

    for i_batch, sample_batched in enumerate(train_dataloader_batches):

        if not is_dataloader and i_batch <= THRESHOLD:
            label, feats, weight = sample_batched
        elif not is_dataloader and i_batch > THRESHOLD and i_epoch > 0:

            label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=1)
        else:
            try:
                label, feats, weight = batch_processor(mini_batch=sample_batched)
            except:
                i_epoch += 1
                label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=1)

                # forward pass
        z_pred = model(feats=feats)

        # backward pass
        E = apply_loss(loss, z_pred, label, weight)
        optimizer.zero_grad()
        E.backward()

        qq1 = time.perf_counter()

        dd3 = qq1 - qq3

        # torch.cuda.synchronize()  # wait for mm to finish
        qq2 = time.perf_counter()

        optimizer.step()

        # torch.cuda.synchronize()  # wait for mm to finish
        qq3 = time.perf_counter()

        loss_val_batch = E.detach().cpu().numpy() * batch_size
        sample_weight_sum_batch = (
            batch_size if weight is None else torch.sum(weight).detach()
        )

        num_batches += 1
        loss_val += loss_val_batch
        loss_val_tb += loss_val_batch
        loss_val_epoch += loss_val_batch
        sample_weight_sum += sample_weight_sum_batch
        sample_weight_sum_tb += sample_weight_sum_batch
        sample_weight_sum_epoch += sample_weight_sum_batch

        if need_to_log_batch(i_batch, logging_options, batch_size):
            log_train_info(
                i_batch=i_batch,
                i_epoch=i_epoch,
                trainer_id=trainer_id,
                start_time=start_time,
                total_loss=loss_val,
                num_batches=num_batches,
                sample_weight_sum=sample_weight_sum,
                batch_size=batch_size,
                lock=lock,
            )
            start_time, loss_val, num_batches, sample_weight_sum = (
                time.time(),
                0.0,
                0,
                0.0,
            )
        if writer is not None and need_to_log_tb(i_batch, logging_options, batch_size):
            log_tb_info_batch(
                writer=writer,
                model=model,
                pred=z_pred,
                label=label,
                optimizer=optimizer,
                logging_options=logging_options,
                iter=total_num_batches * i_epoch + i_batch,
                start_time=start_time_tb,
                trainer_id=trainer_id,
                avg_loss=loss_val_tb / sample_weight_sum_tb,
                lock=lock,
            )
            start_time_tb, loss_val_tb, sample_weight_sum_tb = (time.time(), 0.0, 0.0)

        dd1 = qq2 - qq1
        dd2 = qq3 - qq2

        q1.append(dd2)
        q2.append(dd3)

        if not is_dataloader and i_batch > THRESHOLD:
            label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=2)

    avg_loss = loss_val_epoch / sample_weight_sum_epoch
    return i_batch, avg_loss, q1, q2


def evaluate(model, loss, dataloader, batch_processor, device, batch_size=1024, is_dataloader=True, i_epoch=0):
    model.eval()
    preds = []
    labels = []
    batch_size = batch_size
    loss_val, sample_weight_sum = 0.0, 0.0
    for i_batch, sample_batched in enumerate(dataloader):

        if not is_dataloader and i_batch <= VAL_THRESHOLD:
            label, feats, weight = sample_batched
        elif not is_dataloader and i_batch > VAL_THRESHOLD and i_epoch > 0:
            label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=1)
        else:
            try:
                label, feats, weight = batch_processor(mini_batch=sample_batched)
            except:
                i_epoch += 1
                label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=1)

                # forward pass
        z_pred = model(feats=feats)
        # preds.append(z_pred.detach().cpu().numpy())
        # labels.append(label.detach().cpu().numpy())
        preds += z_pred.detach().cpu().numpy().tolist()
        labels += label.detach().cpu().numpy().tolist()
        E = apply_loss(loss, z_pred, label, weight)
        loss_val += E.detach().cpu().numpy() * batch_size
        sample_weight_sum += (
            batch_size if weight is None else torch.sum(weight).detach().cpu().numpy()
        )

        if not is_dataloader and i_batch > VAL_THRESHOLD:
            label, feats, weight = batch_processor(mini_batch=sample_batched, reverse=2)

            # logger.warning("loss_val: {}, weight_sum {}".format(dataloader, is_dataloader))
    avg_loss = loss_val / sample_weight_sum
    # labels = np.asarray(labels).flatten()
    # preds = np.asarray(preds).flatten()
    try:
        avg_auc = roc_auc_score(labels, preds)
    except Exception:
        idx = np.isfinite(preds)
        avg_auc = roc_auc_score(np.array(labels)[idx], np.array(preds)[idx])

    return avg_loss, labels, preds, avg_auc
