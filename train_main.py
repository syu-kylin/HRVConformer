import os
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
from data_loader import SignalDataset, NormalizeAndToTensor, read_split_data
from model.ConformerNet import confermer_net
from model.FCN import FCN13_HRV_5min
from model.HRVTransformer import hrv_transformer
from model.ResNet import resnet1d50
from utils import NativeScalerWithGradNormCount as NativeScaler
from train_func import train_one_epoch, evaluate
from matrix import auc_binary
from utils import get_structed_log, keep_first_n_line
from postprocessing import postprocessing, train_summary
from plot_figures import plot_curves
from project_init import project_init, setup_logger

logger = logging.getLogger('project_log')

def main(config):

    setup_seed(config.seed)
    misc.init_distributed_mode(config)
    device = torch.device(config.device)

    # Load data
    train_epochs, val_epochs, test_epochs = read_split_data(config.window_length, config.seed_epoch)
    signal_transform = NormalizeAndToTensor(mean=config.mean, std=config.std, 
                                            min=config.min, max=config.max, 
                                            min_max_enable=config.min_max_enable)    
    train_dataset = SignalDataset(train_epochs, 'train', signal_transform)
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
        num_workers=config.num_workers, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(
        val_dataset, batch_size=config.batchsize, sampler=sampler_val,
        num_workers=config.num_workers, pin_memory=True, drop_last=False)
    data_loader_test = DataLoader(
        test_dataset, batch_size=config.batchsize, sampler=sampler_test,
        num_workers=config.num_workers, pin_memory=True, drop_last=False)
    
    # Load model
    # HRVConformerNet
    if config.model_name == 'HrvConformer':
        model = confermer_net(config).to(device)
    # HRVResNet
    elif config.model_name == 'HrvResnet':
        model = resnet1d50(in_channels=1, num_classes=config.n_class, drop_rate=config.dropout).to(device)
    # HRVTransformer
    elif config.model_name == 'HrvTransformer':
        model = hrv_transformer(config).to(device)

    model_without_ddp = model
    print(f'\033[35;1mmodel initialized with {model.__class__.__name__}!\033[0m')

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module

    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, 
                                  betas=(config.beta1, config.beta2), 
                                  eps=config.epsilon)
    loss_func = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler()
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=config.warmup_epoch, max_iters=config.epochs, eta_min=config.lr_min)
    # lr_scheduler = CosineWarmupRestartsScheduler(optimizer, warmup_epochs=config.warmup_epoch, T_0=config.lr_T0, T_mult=config.lr_Tmult, eta_min=config.lr_min, last_epoch=-1)
    # lr_scheduler = None
    config.model_name = model.__class__.__name__
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
    max_auc = 0.0

    wandb.init(project=config.project_name, name=f'{config.run_name}', 
               job_type=config.job_name, group=config.group_name)
    # wandb.config.update(config)
    wandb.watch(model, log='gradients', log_freq=100)   # log gradients every 1000 steps

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
        'eta: {eta}/{eta_total_time}.'
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

        if (epoch % 20 == 0 or (epoch + 1) == config.epochs):
            misc.save_model(
                args=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=scaler, lr_scheduler=lr_scheduler, epoch=epoch) 
            
        model.eval()
        val_stats = evaluate(model, data_loader_val, device)[0]
        val_auc = auc_binary(model, data_loader_val, device)[-1]
        train_auc = auc_binary(model, data_loader_train, device)[-1]
        train_loss, train_acc = train_stats['loss'], train_stats['acc']
        val_loss, val_acc = val_stats['loss'], val_stats['acc']
        step_time, data_time = train_stats['step_time'], train_stats['step_data_time']
        epoch_time = train_stats['epoch_time']
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(iter_time=step_time, data_time=data_time, epoch_time=epoch_time)

        # Taking the exponentially moving average of validation AUC
        if epoch == 0 or epoch == config.start_epoch:
            moving_val_auc = val_auc
        else:
            moving_val_auc = moving_avg(val_auc, moving_val_auc, beta_ema)

        if misc.is_main_process() and (epoch + 1) % config.print_freq == 0 or epoch in (config.start_epoch, config.start_epoch + 1, config.epochs - 1):
            eta = (config.epochs - epoch) * (time.time() - start_time) / (epoch + 1)
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            eta_total_time = (config.epochs - config.start_epoch) * (time.time() - start_time) / (epoch + 1)
            eta_total_time_str = str(datetime.timedelta(seconds=int(eta_total_time)))
            print(log_msg.format(
                epoch + 1, config.epochs, lr=lr, train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc, train_auc=train_auc, val_auc=val_auc,
                epoch_time=train_stats['epoch_time'], step_time=train_stats['step_time'], data_time=train_stats['step_data_time'],
                peak_mem=train_stats['memory'], eta=eta_str, eta_total_time=eta_total_time_str,
            ))
        
        if config.outdir and misc.is_main_process():
            log_stats = {
                'epoch': epoch,'lr': lr,
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_acc': train_acc, 'val_acc': val_acc,
                'train_auc': train_auc, 'val_auc': val_auc, 'moving_val_auc': moving_val_auc,
            }
            with open(config.run_log_fn, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")    
        
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 
                        'train_acc': train_acc, 'val_acc': val_acc,
                        'train_auc': train_auc, 'val_auc': val_auc,
                        'moving_val_auc':moving_val_auc, 'lr': lr, 'epoch': epoch})
            
        # get the best model
        if moving_val_auc > max_auc:
            max_auc = moving_val_auc
            val_acc_bm = val_acc
            train_acc_bm = train_acc
            best_model_epoch = epoch
            val_auc_bm = val_auc
            train_auc_bm = train_auc
            # best_model_path = f"{config.outdir}/model-{config.run_name}.pth"

            if misc.is_main_process():
                torch.save(model_without_ddp.state_dict(), config.best_model_path)

    end_time = time.time()
    training_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(training_time)))
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), model_without_ddp.parameters()))
    metric_logger.synchronize_between_processes()

    # all read write operations should be done in the main process
    if misc.is_main_process():
        msg = ''.join([
            '\n', '='*45, '\n',
            f"Best model found at epoch: {best_model_epoch}/{config.epochs}.\n",
            f"Best model validation AUC {val_auc_bm:.4f}.\n",
            f"Best model validation accuracy {val_acc_bm:.4f}.\n",
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
            'train_auc_bm': train_auc_bm,
            'train_acc_bm': train_acc_bm,
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
        config.save(config.run_config_fn)

    # laod the best model and prepare for postprocessing
    logger.info('\033[35;1mloading best model for postprocessing...\033[0m')
    # best_model_path = f"{config.outdir}/model-{config.run_name}.pth"
    best_model_state_dict = torch.load(config.best_model_path, map_location='cpu', weights_only=True)
    model_without_ddp.load_state_dict(best_model_state_dict)
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), model_without_ddp.parameters()))
    logger.info('finetune best model loaded!')
    my_model = model

    # ! need to check the model type and device for distributed training
    logger.info(type(my_model))
    test_stats = evaluate(my_model, data_loader_test, device)[0]
    fpr_val, tpr_val, val_auc_bm = auc_binary(my_model, data_loader_val, device=device)
    fpr_test, tpr_test, test_auc_bm = auc_binary(my_model, data_loader_test, device=device)
    test_acc_bm = test_stats['acc']
    if misc.is_main_process():
        logger.info(f'Selected model validation AUC: {val_auc_bm:.4f}')
        logger.info(f'Selected model test AUC: {test_auc_bm:.4f}')
        test_auc_bm_dict = {
            'test_acc_bm': test_acc_bm,
            'test_auc_bm': test_auc_bm,
            'val_auc_bm': val_auc_bm,
            'fpr_test': fpr_test,
            'tpr_test': tpr_test,
            'fpr_val': fpr_val,
            'tpr_val': tpr_val,
        }
        run_log = Params(config.log_json_fn)
        run_log.update(test_auc_bm_dict)

        wandb.summary.update({'val_auc_bm': val_auc_bm, 'test_auc_bm': test_auc_bm})

    postprocessing(my_model, val_epochs, data_loader_val, device, config.set_name_val, config)
    postprocessing(my_model, test_epochs, data_loader_test, device, config.set_name_test, config)
    if misc.is_main_process():
        plot_curves(config)
        train_summary(config)



if __name__ == '__main__':

    # ------------------- project initialization -------------------
    # job_name = 'Test'
    # group_name = 'test'

    job_name = 'ModelTest'
    # group_name = 'test'
    # group_name = 'HRVTransformer'
    # group_name = 'HrvResnet'
    group_name = 'HrvConformer'

    # job_name = 'ModelHyperTune'
    # group_name = 'attn_heads'
    # group_name = 'n_layer'
    # group_name = 'kernel_size'
    # group_name = 'patch_size'
    # group_name = 'nuisances'

    config_json_path = project_init(job_name, group_name)
    config = Params(config_json_path)


    # ------------------- run the model -------------------
    def run(config):

        run_name = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(int(time.time())))
        config.run_name = f'{run_name} {config.num_run}'
        # config.run_name = '2025-03-18 16-54-53 1'

        config.outdir = f'{config.parent_dir}/{config.run_name}/'
        os.makedirs(config.outdir, exist_ok=True)
        config.log_fn = f'{config.outdir}/report-{config.run_name}.txt'
        config.run_log_fn = f'{config.outdir}/run_log-{config.run_name}.json'
        config.log_json_fn = f'{config.outdir}/log-{config.run_name}.json'
        config.run_config_fn = f'{config.outdir}/run_config-{config.run_name}.json'
        config.best_model_path = f"{config.outdir}/best_model-{config.run_name}.pth"

        # config.resume = f'{config.outdir}/checkpoint.pth'

        logger = setup_logger(config.log_fn)
        
        config.model_name = 'HrvConformer'
        # config.model_name = 'HrvTransformer'
        # config.model_name = 'HrvResnet'

        config.seed = 32
        config.epochs = 2000
        config.print_freq = 100

        # config.learning_rate = 3e-5
        # config.lr_min = 1e-6
        # config.warmup_epoch = 200
        # config.lr_T0 = 60
        # config.lr_Tmult = 2

        # config.weight_decay = 0.05
        config.dropout = 0.3
        # config.beta1 = 0.85
        # config.beta2 = 0.998

        # # model parameters
        # config.patch_size = 80
        # config.d_model = 144
        # config.num_attention_heads = 6
        # config.n_layer = 3
        # config.conv_kernel_size = 11
        # config.fcn_head_kernel_size = 11

        # config.min_max_enable = True
        # config.batchsize = 1024
        # config.num_workers = 2

        hostname = socket.gethostname()
        if hostname == 'inf-dev71':
            hostname_alias = 'syuLinux'
        elif hostname == 'infantwgb80':
            hostname_alias = 'A10Linux'
        else:
            hostname_alias = hostname
        run_notes = [
            f'{config.model_name} test: n_layer: {config.n_layer}, random seed: {config.seed}, run on {hostname_alias} with copied env.',
            f'beta1: {config.beta1}, d_model: {config.d_model}, fcn_head_kernel_size: {config.fcn_head_kernel_size}, fcn_head_kernel_size: {config.fcn_head_kernel_size}.',
            f'dropout: {config.dropout}, weight decay: {config.weight_decay}, beta2: {config.beta2}, lr min: {config.lr_min:.2e}.',
            f'randomly select 500 epochs for validation (before 20% od dev set: 561h), the rest for training.',
            f'use cosine warmup scheduler, warmup epoch {config.warmup_epoch}, lr: {config.learning_rate}.',
            'batchsize test: 1024, number of workers: 2.',
            'use min-max normalization.',
            'Load dataset from all ANSeR1&2 weak label group and ANSeR1&2 strong label group.',
        ]
        log_notes = '\n'.join(run_notes)
        logger.info(f"\n{'='*45}\n{log_notes}\n{'='*45}\n")
        config.notes = run_notes[:-1]

        config.save(config.run_config_fn)
        main(config)

    # ------------------- tune hyperparameters -------------------
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script")
    parser.add_argument("--param_name", type=str, 
                        help="The name of the hyperparameter to tune.")
    # Argument for hyperparameter values (comma-separated)
    parser.add_argument("--values", type=str, 
                        help="A comma-separated list of values to tune the parameter.")
    args = parser.parse_args()

    def tune_hyperparameters(config, param_name, values):
        i = 0
        for value in values:
            i += 1
            config.num_run = i
            config.dict[param_name] = value
            run(config)

    def tune_multi_hyperparameters(param_dict):
        run_i = 0
        param_names = list(param_dict.keys())
        values = list(param_dict.values())
        for i in range(len(values[0])):
            run_i += 1
            config.num_run = run_i
            for j, param_name in enumerate(param_names):
                config.dict[param_name] = values[j][i]
                # print(f'{param_name}: {values[j][i]}')
            run(config)


    if args.param_name and args.values:
        # Convert comma-separated values to a list
        values = [float(v) if '.' in v else int(v) for v in args.values.split(",")]
        print(f"Hyperparameter tuning for {args.param_name} with values: {values}")
        tune_hyperparameters(config, args.param_name, values)
    else:
        run(config)
    