import os 
import json
import torch
import time as time
import logging
import argparse
import socket

logger = logging.getLogger('project_log')


def get_args():

    parser = argparse.ArgumentParser(description='Project Initialization')
    parser.add_argument('--project_name', type=str, default='HrvConfermer-weak-label', help='Name of the project')
    parser.add_argument('--job_name', type=str, default="Test", help='Name of the job')
    parser.add_argument('--group_name', type=str, default="test", help='Name of the group')
    parser.add_argument('--num_run', type=int, default=1, help='Number of runs for the job')
    parser.add_argument('--outdir', type=str, default='./log', help='Output directory for the project')
    parser.add_argument('--run_name', type=str, default='', help='Name of the run (optional)')
    parser.add_argument('--log_fn', type=str, default='', help='Directory for logs')
    parser.add_argument('--run_config_fn', type=str, default='config.json', help='Run configuration file name (optional)')
    parser.add_argument('--run_log_fn', type=str, default=f'run_log.json', help='Run log file name')
    parser.add_argument('--log_json_fn', type=str, default=f'log_json.json', help='Log JSON file name')
    parser.add_argument('--seed', type=int, default=259, help='Random seed for reproducibility')
    parser.add_argument('--notes', type=str, default='', help='Additional notes for the project')
    parser.add_argument('--best_model_path', type=str, default='', help='Path to save the best model')
    
    parser.add_argument('--wandb_enable', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--save_model', action='store_true', help='Enable saving model checkpoints')
    
    parser.add_argument('--model_name', type=str, default='HrvConfermer', help='Name of the model')
    parser.add_argument('--input_dim', type=int, default=1200, help='Input dimension for the model')
    parser.add_argument('--patch_size', type=int, default=80, help='Patch size for the model')
    parser.add_argument('--d_model', type=int, default=144, help='Dimension of the model')
    parser.add_argument('--attention_type', type=str, default='RelativePositionBias', help="Type of attention module, 'RelativePositionBias' or 'Standard'.")
    parser.add_argument('--fixed_position_embedding', action='store_true', help='Enable fixed position embedding')
    parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of attention heads in the model')
    parser.add_argument('--n_layer', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--conv_kernel_size', type=int, default=11, help='Kernel size for convolutional layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for the model')
    parser.add_argument('--classifier_head', type=str, default='fcn', choices=['fcn', 'mlp_cls', 'mlp_glob_pool'], help='Type of classifier head')
    parser.add_argument('--fcn_head_kernel_size', type=int, default=9, help='Kernel size for FCN head')
    parser.add_argument('--mlp_hid_dim', type=int, default=200, help='Hidden dimension for MLP head')
    parser.add_argument('--ff_dim_factor', type=int, default=4, help='Feed-forward dimension factor')
    parser.add_argument('--n_class', type=int, default=2, help='Number of classes for classification')

    parser.add_argument('--sfreq', type=int, default=4, help='Sampling frequency')
    parser.add_argument('--window_length', type=int, default=5, help='Window length in minutes')
    parser.add_argument('--overlap', type=float, default=0.8, help='Overlap ratio for windows')
    parser.add_argument('--mean', type=float, default=0.5349354123958889, help='Mean for normalization')
    parser.add_argument('--std', type=float, default=0.10205481709220571, help='Standard deviation for normalization')
    parser.add_argument('--min', type=float, default=0.23238983050852707, help='Minimum value for normalization')
    parser.add_argument('--max', type=float, default=0.8566320754717797, help='Maximum value for normalization')
    parser.add_argument('--min_max_enable', action='store_true', help='Enable min-max normalization')
    parser.add_argument('--train_epochs_ratio', type=float, default=1.0, help='Ratio of epochs for training')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')
    parser.add_argument('--seed_epoch', type=int, default=92, help='Epoch to set the random seed')

    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for training')
    parser.add_argument('--print_freq', type=int, default=100, help='Frequency of printing training progress')
    parser.add_argument('--batchsize', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--accum_iter', type=int, default=1, help='Number of iterations to accumulate gradients')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')

    parser.add_argument('--optimizer_name', type=str, default='AdamW', help='Name of the optimizer')
    parser.add_argument('--loss_func', type=str, default='CrossEntropy', help='Loss function to use')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor for the loss function')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--beta1', type=float, default=0.85, help='Beta1 parameter for AdamW optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon parameter for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.998, help='Beta2 parameter for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for the optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--warmup_epoch', type=int, default=500, help='Number of warmup epochs for learning rate scheduler')
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup', help='Learning rate scheduler type')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate for the scheduler')
    parser.add_argument('--lr_Tmult', type=int, default=2, help='Tmult parameter for cosine scheduler')
    parser.add_argument('--lr_T0', type=int, default=60, help='T0 parameter for cosine scheduler')

    parser.add_argument('--init_method', type=str, default='xavier_uniform', help='Initialization method for model weights')
    parser.add_argument('--lr_scheduler1', type=str, default=None, help='First learning rate scheduler (optional)')
    
    parser.add_argument('--dist_on_itp', action='store_true', help='Enable distributed training on ITP')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--dist_eval', action='store_true', help='Enable distributed evaluation')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--dist_url', type=str, default='env://', help='URL for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use for training')

    args = parser.parse_args()

    return args



def setup_logger(log_fn):

    logger = logging.getLogger('project_log')
    logger.setLevel(logging.DEBUG)

    # Determine the global rank (for torchrun and SLURM)
    rank = int(
        os.environ.get("RANK",            # torchrun
        os.environ.get("SLURM_PROCID",    # SLURM
        0))                               # default
    )
    if rank != 0:
        # Disable all logging output for non-zero ranks
        logger.setLevel(logging.CRITICAL + 1)  # Higher than the highest logging level
        return logger
    
    if logger.hasHandlers():  # Avoid adding multiple handlers
        logger.handlers.clear()

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    hostname = socket.gethostname()
    formatter = logging.Formatter(
        f'%(asctime)s | {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    consoleHandler.setFormatter(formatter)              
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(log_fn, mode='w')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    return logger




if __name__ == '__main__':
    
    args = get_args()
