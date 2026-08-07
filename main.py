import os
import sys
import datetime
import argparse
import json
import numpy as np
import random
import time
import socket

import torch
from torch.utils.data import DataLoader
import wandb
import timm.optim as optim_factory
import logging

import misc
from utils import Params, CosineWarmupScheduler, CosineWarmupRestartsScheduler, setup_seed
from data_loader import SignalDataset, NormalizeAndToTensor, ComposeRR, RRAugment
from data_loader import split_baby_independent, read_split_data
from model.ConformerNet import hrvconformer
from utils import NativeScalerWithGradNormCount as NativeScaler
from train_func import train_one_epoch, evaluate
from matrix import auc_binary
from utils import get_structed_log, keep_first_n_line, save_as_json
from postprocessing import postprocessing, train_summary
from plot_figures import plot_curves
from project_init import setup_logger, get_args

logger = logging.getLogger('project_log')

def main(config):

    setup_seed(config.seed)
    if config.distributed:
        misc.init_distributed_mode(config)
    device = torch.device(config.device)

    # Load data
    misc.memory_usage()
    train_epochs, val_epochs, test_epochs = read_split_data(
        config.window_length,
        config.seed_epoch,
        config.train_epochs_ratio,
        data_root=config.data_root,
    )
    signal_transform = NormalizeAndToTensor(mean=config.mean, std=config.std, 
                                            min=config.min, max=config.max, 
                                            min_max_enable=config.min_max_enable) 
    train_dataset = SignalDataset(train_epochs, 'train', signal_transform)     # train set add augmentation
    val_dataset = SignalDataset(val_epochs, 'validation', signal_transform)
    test_dataset = SignalDataset(test_epochs, 'test', signal_transform)

    if config.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if config.dist_eval:
            if len(val_dataset) % num_tasks != 0 or len(test_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
            sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_train = DataLoader(
        train_dataset, batch_size=config.batchsize, sampler=sampler_train,
        num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    data_loader_val = DataLoader(
        val_dataset, batch_size=config.batchsize, sampler=sampler_val,
        num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)
    data_loader_test = DataLoader(
        test_dataset, batch_size=config.batchsize, sampler=sampler_test,
        num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)
    misc.memory_usage()

    # Load model
    model = hrvconformer(config).to(device)
    model_without_ddp = model
    logger.info(f'\033[35;1mmodel initialized with {model.__class__.__name__}!\033[0m')

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module

    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, 
                                  betas=(config.beta1, config.beta2), 
                                  eps=config.epsilon)
    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = NativeScaler()
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=config.warmup_epoch, max_iters=config.epochs, eta_min=config.lr_min)
    config.model_name = model_without_ddp.__class__.__name__
    config.loss_func = loss_func.__class__.__name__
    config.optimizer_name = optimizer.__class__.__name__
    config.lr_scheduler = lr_scheduler.__class__.__name__

    # whether to resume training
    misc.load_model(args=config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=scaler, lr_scheduler=lr_scheduler)
    if config.resume:
        keep_first_n_line(config.run_log_fn, config.start_epoch)

    # Training
    # Expentionaly moving average for training average auc
    n_beta_ema = 5
    beta_ema = 1 - (1/n_beta_ema)
    # Taking the exponentially moving average of validation AUC
    def moving_avg(value, moving_value, beta):
        moving_value = (1-beta)*value + beta*moving_value
        return moving_value
    max_auc, min_loss = 0.0, 100
    if config.wandb_enable:
        wandb.init(project=config.project_name, name=f'{config.run_name}', 
                job_type=config.job_name, group=config.group_name)

    # log the training process of each epoch
    space_fmt = ':' + str(len(str(config.epochs))) + 'd'
    log_msg = [
        'Epoch: [{0' + space_fmt + '}/{1}]',
        'lr: {lr:.2e}',
        'train_loss: {train_loss:.4f}',
        'val_loss: {val_loss:.4f}',
        'train_acc: {train_acc:.4f}',
        'val_acc: {val_acc:.4f}',
        'train_auc: {train_auc:.4f}',
        'val_auc: {val_auc:.4f}',
        'epoch_time: {epoch_time:.2f}s/epoch',
        'iter time: {step_time:.3f}s/step',
        'data time: {data_time:.3f}s/step',
        'peak mem: {peak_mem:.3f}GB',
        'eta: {eta}<{elapsed_time}.'
    ]
    log_msg = ', '.join(log_msg)

    # get epoch metric logger for summary
    metric_logger = misc.MetricLogger(delimiter="s, ")
    metric_logger.add_meter('iter_time', misc.SmoothedValue(window_size=6, fmt='{avg:.4f}s ({global_avg:.4f})'))
    metric_logger.add_meter('data_time', misc.SmoothedValue(window_size=6, fmt='{avg:.4f}s ({global_avg:.4f})'))
    metric_logger.add_meter('epoch_time', misc.SmoothedValue(window_size=6, fmt='{avg:.4f}s ({global_avg:.4f})'))

    time_now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(time.time())))
    start_time = time.time()
    logger.info(f"\033[35;1m{time_now} Start training for {config.epochs} epochs from epoch {config.start_epoch}.\033[0m")

    for epoch in range(config.start_epoch, config.epochs):

        model.train()
        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train, loss_func, optimizer, scaler, device, epoch, config)
        lr_scheduler.step()

        if config.save_model and (epoch % 20 == 0 or (epoch + 1) == config.epochs):
            misc.save_model(
                args=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=scaler, lr_scheduler=lr_scheduler, epoch=epoch) 
            
        model.eval()
        val_stats = evaluate(model, data_loader_val, device)
        val_auc = auc_binary(model, data_loader_val, device)
        train_auc = auc_binary(model, data_loader_train, device)

        train_loss, train_acc = train_stats['loss'], train_stats['acc']
        train_grad_norm = train_stats['grad_norm']
        val_loss, val_acc_seg, val_acc_epoch = val_stats['loss'], val_stats['acc'], val_stats['acc_epoch']
        step_time, data_time = train_stats['step_time'], train_stats['step_data_time']
        epoch_time = train_stats['epoch_time']
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(iter_time=step_time, data_time=data_time, epoch_time=epoch_time)

        # Taking the exponentially moving average of validation AUC
        if epoch == 0 or epoch == config.start_epoch:
            moving_val_auc = val_auc
            moving_val_loss = val_loss
        else:
            moving_val_auc = moving_avg(val_auc, moving_val_auc, beta_ema)
            moving_val_loss = moving_avg(val_loss, moving_val_loss, beta_ema)

        # print the training process
        if misc.is_main_process() and (epoch + 1) % config.print_freq == 0 or epoch in (config.start_epoch, config.start_epoch + 1, config.epochs - 1):
            eta_end = (config.epochs - (epoch + 1)) * (time.time() - start_time) / (epoch + 1)
            eta_end_str = str(datetime.timedelta(seconds=int(eta_end)))
            time_elapsed = (time.time() - start_time)
            time_elapsed_str = str(datetime.timedelta(seconds=int(time_elapsed)))
            logger.info(log_msg.format(
                epoch + 1, config.epochs, lr=lr, train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc_seg, train_auc=train_auc, val_auc=val_auc,
                epoch_time=train_stats['epoch_time'], step_time=train_stats['step_time'], data_time=train_stats['step_data_time'],
                peak_mem=train_stats['memory'], eta=eta_end_str, elapsed_time=time_elapsed_str,
            ))
        
        # log the training process to file and wandb
        if config.outdir and misc.is_main_process():
            log_stats = {
                'epoch': epoch,'lr': lr, 'grad_norm': train_grad_norm, 'clipped_norm': train_stats['clipped_norm'],
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_acc': train_acc, 'val_acc': val_acc_seg, 'val_acc_epoch': val_acc_epoch,
                'train_auc': train_auc, 'val_auc': val_auc, 'moving_val_auc': moving_val_auc,
            }
            with open(config.run_log_fn, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")    
            if config.wandb_enable:
                # log the training process to wandb
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 
                            'train_acc': train_acc, 'val_acc': val_acc_seg, 'val_acc_epoch': val_acc_epoch,
                            'train_auc': train_auc, 'val_auc': val_auc,
                            'moving_val_auc':moving_val_auc, 'lr': lr, 'epoch': epoch})
            
        # get the best model
        # if moving_val_loss < min_loss:
        if moving_val_auc > max_auc:
            # min_loss = moving_val_loss
            max_auc = moving_val_auc
            val_acc_bm = val_acc_seg
            val_acc_epoch_bm = val_acc_epoch
            train_acc_bm = train_acc
            val_auc_bm = val_auc
            train_auc_bm = train_auc
            train_loss_bm = train_loss
            best_model_epoch = epoch

            if misc.is_main_process():
                torch.save(model_without_ddp.state_dict(), config.best_model_path)

    end_time = time.time()
    training_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(training_time)))
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), model_without_ddp.parameters()))
    metric_logger.synchronize_between_processes()
    misc.memory_usage()

    # all read write operations should be done in the main process
    if misc.is_main_process():
        msg = ''.join([
            '\n', '='*45, '\n',
            f"Best model found at epoch: {best_model_epoch}/{config.epochs}.\n",
            f"Best model validation AUC {val_auc_bm:.4f}.\n",
            f"Best model validation accuracy on segment level {val_acc_bm:.4f}.\n",
            f"Best model validation accuracy on epoch level {val_acc_epoch_bm:.4f}.\n",
            f"Best model training AUC {train_auc_bm:.4f}.\n", 
            f"Best mdoel training accuracy {train_acc_bm:.4f}.\n\n",
            '='*45, '\n',
            '\033[35;1m\nEnd of training. {} Cost {} ({:.2f}h).\n\033[0m'.format(
            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 
            total_time_str, training_time/3600),
            '\033[32;1mAverage training speed: {}s, peak memory: {:.3f}GB.\033[0m\n'.format(metric_logger, train_stats['memory']),
        ])
        logger.info(msg)

        best_model_attr = {
            'epoch_bm': best_model_epoch,
            'val_auc_bm': val_auc_bm,
            'val_acc_bm': val_acc_bm,
            'val_acc_epoch_bm': val_acc_epoch_bm,
            'train_auc_bm': train_auc_bm,
            'train_acc_bm': train_acc_bm,
            'train_loss_bm': train_loss_bm,
            'moving_val_auc_bm': max_auc,
            'training_time': training_time,
            'model_capacity': num_parameters,
        }
        train_dict = get_structed_log(config.run_log_fn)
        train_dict.update(best_model_attr)
        if os.path.isfile(config.log_json_fn):
            run_log = Params(config.log_json_fn)
            run_log.update(train_dict)
        else:
            with open(config.log_json_fn, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_dict) + "\n")
        logger.info('training log data converted!')

        # update the run config file
        save_as_json(vars(config), config.run_config_fn)
        logger.info(f'run config saved to {config.run_config_fn}.')

    # load the best model and prepare for final evaluation
    logger.info('\033[35;1mloading best model for final evaluation...\033[0m')
    best_model_state_dict = torch.load(config.best_model_path, map_location='cpu', weights_only=True)
    model_without_ddp.load_state_dict(best_model_state_dict)
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), model_without_ddp.parameters()))
    logger.info('finetune best model loaded!')
    my_model = model

    val_stats = evaluate(my_model, data_loader_val, device)
    test_stats = evaluate(my_model, data_loader_test, device)
    val_auc_states = auc_binary(my_model, data_loader_val, device=device, epoch_aggregation=True, verbose=True)
    test_auc_states = auc_binary(my_model, data_loader_test, device=device, epoch_aggregation=True, verbose=True)
    
    test_acc_bm, test_acc_epoch = test_stats['acc'], test_stats['acc_epoch']
    val_acc_bm, val_acc_epoch = val_stats['acc'], val_stats['acc_epoch']
    if misc.is_main_process():
        message = ''.join([
            '\n', '='*45, '\n',
            "Selected model validation AUC on segment level: {:.4f}.\n".format(val_auc_states['roc_auc_seg']),
            "Selected model validation AUC on epoch level: {:.4f}.\n".format(val_auc_states['roc_auc_epoch']),
            "Selected model validation accuracy on segment level {:.4f}.\n".format(val_stats['acc']),
            "Selected model validation accuracy on epoch level {:.4f}.\n\n".format(val_stats['acc_epoch']),
            "Selected model test AUC on segment level: {:.4f}.\n".format(test_auc_states['roc_auc_seg']),
            "Selected model test AUC on epoch level: {:.4f}.\n".format(test_auc_states['roc_auc_epoch']),
            "Selected model test accuracy on segment level {:.4f}.\n".format(test_stats['acc']),
            "Selected model test accuracy on epoch level {:.4f}.\n\n".format(test_stats['acc_epoch']),
            '='*45, '\n',
        ])
        logger.info(message)
        best_model_dict = {
            'val_acc_bm': val_acc_bm,
            'val_acc_epoch_bm': val_acc_epoch,
            'val_auc_seg_bm': val_auc_states['roc_auc_seg'],
            'val_auc_epoch_bm': val_auc_states['roc_auc_epoch'],
            'fpr_val_seg': val_auc_states['fpr_seg'],
            'tpr_val_seg': val_auc_states['tpr_seg'],
            'fpr_val_epoch': val_auc_states['fpr_epoch'],
            'tpr_val_epoch': val_auc_states['tpr_epoch'],
            'test_acc_bm': test_acc_bm,
            'test_acc_epoch': test_acc_epoch,
            'test_auc_seg': test_auc_states['roc_auc_seg'],
            'test_auc_epoch': test_auc_states['roc_auc_epoch'],
            'fpr_test_seg': test_auc_states['fpr_seg'],
            'tpr_test_seg': test_auc_states['tpr_seg'],
            'fpr_test_epoch': test_auc_states['fpr_epoch'],
            'tpr_test_epoch': test_auc_states['tpr_epoch'],
        }
        run_log = Params(config.log_json_fn)
        run_log.update(best_model_dict)
        if config.wandb_enable:
            wandb.summary.update({'val_auc_bm': val_auc_bm, 'test_auc_bm': test_auc_states['roc_auc_seg'],
                                  'val_auc_bm_epoch': val_auc_states['roc_auc_epoch'], 'test_auc_bm_epoch': test_auc_states['roc_auc_epoch'],
                                  'val_acc_bm': val_acc_bm, 'test_acc_bm': test_acc_bm,
                                  'val_acc_epoch': val_acc_epoch, 'test_acc_epoch': test_acc_epoch})

    if misc.is_main_process():
        plot_curves(config)
        train_summary(config)

    if config.distributed:
        misc.end_with_cleanup()

if __name__ == '__main__':

    config = get_args()

    if not config.run_name:
        config.run_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config.outdir = os.path.join(config.outdir, config.job_name, config.group_name, config.run_name)
    os.makedirs(config.outdir, exist_ok=True)
    config.log_fn = os.path.join(config.outdir, f'report-{config.run_name}.txt')
    config.run_log_fn = os.path.join(config.outdir, f'run_log-{config.run_name}.json')
    config.log_json_fn = os.path.join(config.outdir, f'log-{config.run_name}.json')
    config.run_config_fn = os.path.join(config.outdir, f'run_config-{config.run_name}.json')
    config.best_model_path = os.path.join(config.outdir, f'best_model-{config.run_name}.pth')

    # config.resume = f'{config.outdir}/checkpoint.pth'

    logger = setup_logger(config.log_fn)

    main(config)
