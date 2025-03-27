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

import numpy as np
import torch

import logging

from utils import Params
from project_init import project_init
import misc


@torch.no_grad()
def evaluate(model, dataloader, device):
    ''' Evaluate the model for each batch.
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
    model.eval()
    for batch in metric_logger.log_every(dataloader, 50, header):
        data, label = batch
        data = data.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda'):
            out = model(data)
            loss = loss_fn(out, label)

        _, pred = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(pred == label).item() / len(pred))
        metric_logger.update(loss=loss, acc=acc)

        y_true_epoch.append(label)
        y_pred_epoch.append(pred)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return metric, y_true_epoch, y_pred_epoch

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

    accum_iter = args.accum_iter
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, 50, header)):
        data, label = batch
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
        grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        _, pred = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(pred == label).item() / len(pred))
        metric_logger.update(loss=loss_value, acc=acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == "__main__":

    config_json_path = project_init()
    config = Params(config_json_path)
