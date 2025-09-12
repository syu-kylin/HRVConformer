# ----------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# Deit: https://github.com/facebookresearch/deit
# ----------------------------------------------

import os
import math
import sys
import copy
import time as time
from typing import Iterable
from collections import defaultdict

import numpy as np
import torch

import logging

from utils import Params
import misc


@torch.no_grad()
def evaluate(model, dataloader, device, return_attention=False):
    ''' Evaluate the model for each batch. This function sutomatically compatable with distributed or non-distributed evl.
    Args: 
        model: (torch.nn.Module) the neural network.
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch.
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data.
    '''
    # set model to eval model.
    loss_fn = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    y_pred_epoch, y_true_epoch = [], []
    all_attn_weight = []
    # Collect predicted probabilities and labels by epoch_id
    epoch_probs = defaultdict(list)
    epoch_preds = defaultdict(list)
    epoch_labels = defaultdict(list)

    model.eval()
    for batch in metric_logger.log_every(dataloader, 50, header):
        data, label, file_ids = batch
        data = data.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda'):
            if return_attention:
                out, attn_weight = model(data, return_attention=True)
                all_attn_weight.append(attn_weight)
            else:
                out = model(data)
            loss = loss_fn(out, label)

        preds = torch.argmax(out, dim=1)     # predicted label per segment, faster and numerically more stable than using softmax output
        probs = torch.softmax(out, dim=1)[:, 1]
        # _, preds = torch.max(out, dim=1)

        # calculate accuracy on the segment level
        acc_seg = torch.tensor(torch.sum(preds == label).item() / len(preds))

        # calculate accuracy on the file level
        # Aggregate probabilities by file_id (epoch_id)
        for prob, label, pred, fid in zip(probs.cpu(), label.cpu(), preds.cpu(), file_ids):
            epoch_probs[fid].append(prob.item())
            epoch_preds[fid].append(pred.item())
            epoch_labels[fid].append(label.item())

        metric_logger.update(loss=loss, acc=acc_seg)
    
    # aggregate the predictions and labels on the epoch level
    labels_epoch_level = []
    preds_epoch_level = []
    for fid in epoch_probs.keys():
        majority_vote_pred = int(sum(epoch_preds[fid]) >= len(epoch_preds[fid]) / 2)
        preds_epoch_level.append(majority_vote_pred)

        label_one_epoch = set(epoch_labels[fid])
        assert len(label_one_epoch) == 1, f"Multiple labels found for file_id {fid}: {label_one_epoch}"
        labels_epoch_level.append(list(label_one_epoch)[0])
    assert len(labels_epoch_level) == len(preds_epoch_level), "Mismatch in lengths of labels and predictions"
    
    # calculate accuracy on the epoch level
    acc_epoch = np.sum(np.array(labels_epoch_level) == np.array(preds_epoch_level)) / len(labels_epoch_level)
    metric_logger.update(acc_epoch=acc_epoch)

    # gather the stats from all processes if distributed (otherwise, it does nothing)
    metric_logger.synchronize_between_processes()
    metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_attention:
        return metric, all_attn_weight
    return metric


def train_one_epoch(model: torch.nn.Module, dataloader: Iterable, loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer, loss_scaler, device: torch.device,
                     epoch: int, args=None):
    ''' 
    Train the model for each batch.
    '''

    # set model to training model.
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Train:'

    all_attn_weight = []
    accum_iter = args.accum_iter
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, 50, header)):
        data, label, _ = batch
        data = data.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda'):
            out = model(data)
            loss = loss_fn(out, label)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        grad_norm, clipped_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad_norm, parameters=model.parameters(), create_graph=False)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        _, pred = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(pred == label).item() / len(pred))
        metric_logger.update(loss=loss_value, acc=acc, grad_norm=grad_norm, clipped_norm=clipped_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



