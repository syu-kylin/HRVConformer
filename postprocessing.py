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
from project_init import project_init
import misc

logger = logging.getLogger('project_log')

def majority_vote(pred_epochs):
    # print(f'pred_epochs: {pred_epochs}')
    calss_counts = np.bincount(pred_epochs)
    final_pred = calss_counts.argmax()
    
    return int(final_pred)

def epoch_check(epoch_struct):

    N_files = len(epoch_struct)
    file_ids, n_epochs = [], []
    grades, grades_binary = [], []
    for i in range(N_files):
        file_id = str(epoch_struct[i]['file_id'])
        grade = int(epoch_struct[i]['EEG_grade'])
        num_epoch_one_file = epoch_struct[i]['n_epochs']
        # file_id = epoch_struct[i]['file_id'][0]
        # grade = epoch_struct[i]['EEG_grade'][0][0]
        # num_epoch_one_file = epoch_struct[i]['n_epochs'][0][0]
        
        n_epochs.append(num_epoch_one_file)
        file_ids.append(file_id)
        grades.append(grade)
        
        # if grade in (0,1):
        #     grade = 0
        # elif grade in (2,3,4):
        #     grade = 1

        grades_binary.append(grade)
        
    # annotation = {'file_id': file_ids, 'grade': grades}
    # anno_df = pd.DataFrame(annotation)
    # anno_df.to_csv('annotaion_train.csv', index=False)

    return n_epochs, file_ids, grades, grades_binary

def postprocessing(my_model, val_epochs, val_generator, device, set_name, config):
    """  
    This functon is a post-processing procedure after training. It uses a majority 
    vote method to generate a prediction for each one-hour epoch and validation 
    accuracy after post-processing. It also summarize the validation results of this 
    run into log file.
    Args:
          my_model: (nn.modules) the best model selected in the train.
          val_epochs: (numpy struct) the validation rr epochs.
          val_generator: (dataloder) the generater of the validation dataloder.
          loss_func: (nn.loss_func) the loss function same as the train.
          set_name: (str) "validation" or "test".
          param: (class of Params) the parameters of run config.
    """

    # Load the run log json
    param_log = Params(config.log_json_fn)

    if set_name == 'validation':
        val_var = 'val'
        val_str = 'validation'
        val_auc = param_log.val_auc_bm
    elif set_name == 'test':
        val_var = 'test'
        val_str = 'test'
        val_auc = param_log.test_auc_bm

    val_stats, label_epoch, pred_epoch = evaluate(my_model, val_generator, device)
    val_acc_bm = val_stats['acc']
    preds_arr = torch.cat(pred_epoch).cpu().numpy()
    labels_arr = torch.cat(label_epoch).cpu().numpy()
    
    n_epochs, file_ids_val, grades_orig, grades_bin = epoch_check(val_epochs)
    
    final_preds, verify_labels = [], []
    seg_start = 0
    for x in n_epochs:
        seg_end = seg_start + x
        pred_one_file = preds_arr[seg_start:seg_end]
        label_one_file = labels_arr[seg_start:seg_end]
    
        final_pred_one_file = majority_vote(pred_one_file)
        final_label_one_file = majority_vote(label_one_file)
    
        final_preds.append(final_pred_one_file)
        verify_labels.append(final_label_one_file)
        seg_start = seg_end
    
    val_acc_post = sum(np.array(final_preds)==np.array(verify_labels))/len(verify_labels)
    if config.distributed:
        val_acc_post = misc.all_reduce_mean(val_acc_post)

    if misc.is_main_process():
        val_bm_dict = {
            f'{val_var}_acc_post': val_acc_post,
            f'{val_var}_acc_bm': val_acc_bm,
        }

        # Add to run log json
        param_log.update(val_bm_dict)
        
        message_terminal = ''.join([
            '\n', '-'*45, '\n',
            '\033[1mPostprocessing\033[0m\n\n',
            '\033[1m{} set file_ids:\n\033[0m{}\n\n'.format(val_str, file_ids_val),
            '\033[1m{} accuracy after postprocessing:\033[0m{:.4f}\n'.format(val_str, val_acc_post),
            '\033[1m{} accuracy before postprocessing:\033[0m{:.4f}\n'.format(val_str, val_acc_bm),
            '\033[1m{0:<20}:\033[0m {1:.4f}\n'.format(f'{val_str} auc', val_auc),
            '\033[1m{0:<20}:\033[0m {1}\n'.format('final predictions', final_preds),
            '\033[1m{0:<20}:\033[0m {1}\n'.format('true labels', grades_bin),
            '\033[1m{0:<20}:\033[0m {1}\n'.format('verified labels', verify_labels),
            '_'*70, '\n\n\n',
        ])
        logger.info(message_terminal)
        
        if bool(config.wandb_enable) and set_name == 'validation':
            wandb.summary[f'{val_var}_acc_post'] = val_acc_post
            wandb.finish()
    


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
        'test accuracy after postprocessing:{:.4f}\n'.format(param_log.test_acc_post),
        'test accuracy before postprocessing: {:.4f}\n'.format(param_log.test_acc_bm),
        'test AUC: {:.4f}\n\n'.format(param_log.test_auc_bm),

        'validation accuracy after postprocessing:{:.4f}\n'.format(param_log.val_acc_post),
        'validation accuracy before postprocessing: {:.4f}\n'.format(param_log.val_acc_bm),
        'validation AUC: {:.4f}\n\n'.format(param_log.val_auc_bm),

        'highest train AUC: {:.4f}\n'.format(max(param_log.train_auc)),
        'highest train ACC: {:.4f}\n'.format(max(param_log.train_acc)),
        'lowest train loss: {:.4f}\n'.format(min(param_log.train_loss)),
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
        'test_acc_post': param_log.test_acc_post,
        'test_acc_bef': param_log.test_acc_bm,
        'test_auc_bm': param_log.test_auc_bm,
        'val_acc_post': param_log.val_acc_post,
        'val_acc_bef': param_log.val_acc_bm,
        'val_auc_bm': param_log.val_auc_bm,
        'highest train auc': max(param_log.train_auc),
        'highest train acc': max(param_log.train_acc),
        'lowest train loss': min(param_log.train_loss),
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
    fn = f'{config.parent_dir}/train_summary_{config.group_name}.csv'
    if os.path.isfile(fn):
        run_summary_df.to_csv(fn, mode='a', header=False, index=False)
    else:
        run_summary_df.to_csv(fn, index=False)

    logger.info("\033[35;1mrun summary file saved!\033[0m")
    # return train_summary


if __name__ == '__main__':

    job_name = 'ModelTest'
    group_name = 'HrvConformer'
    run_id = '2025-03-17 19-56-45 1'

    config_json_path = f'./log/{job_name}/{group_name}/{run_id}/run_config-{run_id}.json'
    config = Params(config_json_path)
    
    train_summary(config)