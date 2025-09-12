import os
import numpy as np
import pandas as pd
import torch
import wandb
import time as timer
import datetime
import logging

from utils import Params
from train_func import evaluate
from project_init import get_args
import misc

logger = logging.getLogger('project_log')



def train_summary(config):
    """  
    Summarize this run results and config information into log file.
    (WARNING: This funtion is the last steps of )
    Args:
        config: the config parameters of this run.
    """
    run_time_end = timer.strftime('%Y-%m-%d %H-%M-%S',timer.localtime(int(timer.time())))
    param_log = Params(config.log_json_fn)

    data_norm_method = 'min-max' if config.min_max_enable else 'standard'
    warmup_epochs = f'{config.warmup_epoch}/{config.lr_T0}' if config.lr_scheduler == 'CosineWarmupRestartsScheduler' else config.warmup_epoch
    run_notes_console = '\n'.join(config.notes)
    run_notes_summary = ' '.join(config.notes)

    message = ''.join([
        '\n', '-'*45, '\n', f'run name: {config.run_name}\n',
        'Training result summary:\n\n',
        'test accuracy after postprocessing:{:.4f}\n'.format(param_log.test_acc_epoch),
        'test accuracy before postprocessing: {:.4f}\n'.format(param_log.test_acc_bm),
        'test AUC on epoch level: {:.4f}\n'.format(param_log.test_auc_epoch),
        'test AUC on segment level: {:.4f}\n\n'.format(param_log.test_auc_seg),

        'validation accuracy after postprocessing:{:.4f}\n'.format(param_log.val_acc_epoch_bm),
        'validation accuracy before postprocessing: {:.4f}\n'.format(param_log.val_acc_bm),
        'validation AUC on epoch level: {:.4f}\n'.format(param_log.val_auc_epoch_bm),
        'validation AUC on segment level: {:.4f}\n\n'.format(param_log.val_auc_seg_bm),

        'highest/bm train AUC: {:.4f}/{:.4f}\n'.format(max(param_log.train_auc), param_log.train_auc_bm),
        'highest/bm train ACC: {:.4f}/{:.4f}\n'.format(max(param_log.train_acc), param_log.train_acc_bm),
        'lowest/bm train loss: {:.4f}/{:.4f}\n'.format(min(param_log.train_loss), param_log.train_loss_bm),
        'best model acquired from epoch: {}/{}\n\n'.format(param_log.epoch_bm, config.epochs),
        
        'Training parameters:\n',
        f'group_name: {config.group_name}\n',
        f'job_name: {config.job_name}\n',
        'Learning rate: {:.2g}\n'.format(config.learning_rate),
        'weight decay: {}\n'.format(config.weight_decay),
        f'optimizer:{config.optimizer_name}\n',
        "betas: ({:.4f}, {:.4f})\n".format(config.beta1, config.beta2),
        "epsilon: {}\n".format(config.epsilon),
        'lr_scheduler: {}\n'.format(config.lr_scheduler), 
        'warmup epochs: {}\n'.format(warmup_epochs),
        'warmup restart T0/T_mul: {}/{}\n'.format(config.lr_T0, config.lr_Tmult),
        'init method: {}\n'.format(config.init_method),
        f'batchsize:{config.batchsize}\n',
        'number of workers: {}\n'.format(config.num_workers),
        f'seed_epoch: {config.seed_epoch}\nrandom seed: {config.seed}\n',
        'epochs:{}\n'.format(config.epochs),
        f'norm_method: {data_norm_method}\n',
        f'patch size: {config.patch_size/config.sfreq}s\n',
        f'convolution kernel size: {config.conv_kernel_size}\n',
        
        f'model training with: {config.model_name}\n',
        f'model number of parameters: {param_log.model_capacity/1e6:.4f}M\n',
        f'Data sample frequency: {config.sfreq} Hz\n',
        f'Window length: {config.window_length} min\n',
        f'Overlap: {config.overlap*100}%\n\n',
        f'Run notes: {run_notes_console}\n\n',
        f'training time: {str(datetime.timedelta(seconds=int(param_log.training_time)))} ({param_log.training_time/3600:.2f} h)\n\n',
        '\033[35;1mEnd of this run {}\033[0m\n'.format(run_time_end),
        '-'*45, '\n',
    ])
    logger.info(message)

    train_summary = {
        'run_name': config.run_name,
        'test_acc_post': param_log.test_acc_epoch,
        'test_acc_bef': param_log.test_acc_bm,
        'test_auc_bm': param_log.test_auc_seg,
        'test_auc_bm_epoch': param_log.test_auc_epoch,
        'val_acc_post': param_log.val_acc_epoch_bm,
        'val_acc_bef': param_log.val_acc_bm,
        'val_auc_bm': param_log.val_auc_seg_bm,
        'val_auc_bm_epoch': param_log.val_auc_epoch_bm,
        'train auc(bm/highest)': '{:.4f}/{:.4f}'.format(max(param_log.train_auc), param_log.train_auc_bm),
        'train acc(bm/highest)': '{:.4f}/{:.4f}'.format(max(param_log.train_acc), param_log.train_acc_bm),
        'train loss(bm/lowest)': '{:.4f}/{:.4f}'.format(min(param_log.train_loss), param_log.train_loss_bm),
        'best model epoch': '{}/{}'.format(param_log.epoch_bm, config.epochs), 
        'architecture': config.model_name,
        'epochs': config.epochs,
        'optimizer': config.optimizer_name,
        'learning rate': '{:.2e}'.format(config.learning_rate),
        'weight decay': config.weight_decay,
        'beta_1': config.beta1,
        'beta_2': config.beta2,
        'epsilon': config.epsilon,
        'warmup epoch': warmup_epochs,
        'lr_scheduler': config.lr_scheduler,
        'n_layer': config.n_layer,
        'd_model': config.d_model,
        'num_attention_heads': config.num_attention_heads,
        'conv_kernel_size': config.conv_kernel_size,
        'classifier head': config.classifier_head,
        'fcn_head_kernel_size': config.fcn_head_kernel_size,
        'dropout': config.dropout,
        'weight_decay': config.weight_decay,
        'patch size': f'{int(config.patch_size/config.sfreq)}s',
        'window length': f'{config.window_length}min',
        'init method': config.init_method,
        'batchsize': config.batchsize,
        'seed_epoch': config.seed_epoch,
        'torch_seed': config.seed,
        'data_norm_method': data_norm_method,
        'train_duration': '{:.2f}h'.format(param_log.training_time/3600),
        'notes': run_notes_summary,
    }
    
    run_summary_df = pd.Series(train_summary).to_frame().T
    fn = f'./log/{config.job_name}/{config.group_name}/train_summary_{config.group_name}.csv'
    if os.path.isfile(fn):
        run_summary_df.to_csv(fn, mode='a', header=False, index=False)
    else:
        run_summary_df.to_csv(fn, index=False)

    logger.info("\033[35;1mrun summary file saved!\033[0m")
    # return train_summary


if __name__ == '__main__':

    job_name = 'ModelTest'
    group_name = 'HRVConformer'
    run_id = '20250626_152336_5'

    config_json_path = f'./log/{job_name}/{group_name}/{run_id}/run_config-{run_id}.json'
    config = Params(config_json_path)
    
    train_summary(config)