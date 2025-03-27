

import json
import os
import torch
import numpy as np
import random
import time
from collections import deque, defaultdict
from torch import distributed as dist
from misc import get_rank

def setup_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, warmup, max_iters, eta_min=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.eta_min = eta_min
        super().__init__(optimizer)
        
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        # return [max(base_lr * lr_factor, self.eta_min) for base_lr in self.base_lrs]
        return [base_lr * lr_factor if self.last_epoch <= self.warmup else max(self.eta_min, base_lr * lr_factor) for base_lr in self.base_lrs]
    
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
    def state_dict(self):

        return {
            'warmup': self.warmup,
            'max_num_iters': self.max_num_iters,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict):
        
        self.warmup = state_dict['warmup']
        self.max_num_iters = state_dict['max_num_iters']
        self.eta_min = state_dict['eta_min']
        self.last_epoch = state_dict['last_epoch']



class CosineWarmupRestartsScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        """
        Combines a warm-up period with CosineAnnealingWarmRestarts.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for warm-up.
            T_0 (int): Number of iterations for the first restart in cosine annealing.
            T_mult (int): Factor by which T_0 is multiplied after each restart. Default: 1.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_T_max = T_0  # Initial cycle length
        self.cycle_epoch_start = warmup_epochs  # Track start of the current cosine cycle
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the updated learning rate for each parameter group.
        """
        # Warm-up phase
        if self.last_epoch < self.warmup_epochs:
            lr_factor = self.last_epoch / self.warmup_epochs

        # Cosine annealing with restarts
        else:
            # Calculate the epoch within the current cosine cycle
            epoch_in_cycle = self.last_epoch - self.cycle_epoch_start

            # If the cycle is complete, restart
            if epoch_in_cycle >= self.current_T_max:
                # Restart the cycle
                self.cycle_epoch_start = self.last_epoch  # Update start of new cycle
                epoch_in_cycle = 0
                self.current_T_max *= self.T_mult  # Increase cycle length by T_mult

            # Compute cosine annealing factor within the current cycle
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_cycle / self.current_T_max))

        # Apply eta_min to ensure the learning rate does not drop below it
        # return [max(base_lr * lr_factor, self.eta_min) for base_lr in self.base_lrs]
        return [base_lr * lr_factor if self.last_epoch <= self.warmup_epochs else max(self.eta_min, base_lr * lr_factor) for base_lr in self.base_lrs]
    
    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'current_T_max': self.current_T_max,
            'cycle_epoch_start': self.cycle_epoch_start,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict['warmup_epochs']
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.current_T_max = state_dict['current_T_max']
        self.cycle_epoch_start = state_dict['cycle_epoch_start']
        self.last_epoch = state_dict['last_epoch']

class NativeScalerWithGradNormCount:
    '''
    This class is a wrapper for the PyTorch native GradScaler class. It adds the ability to return the gradient norm
    after the optimizer step. loss.backward() step will be executed here.
    This class provided mixed precision training with gradient scaling and gradient clipping and scaler state management
    also returns the gradient norm.
    '''

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        '''
        Returns the state dict of the scaler. This is useful for saving the state of the scaler to disk for resuming training.
        '''
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        '''
        Loads the state dict of the scaler. This is useful for resuming training from a saved state.
        '''
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    '''
    Returns the norm of the gradients of the parameters. This is useful for logging the gradient norm during training.
    '''
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)   # load json file 
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    params.dict['new_member'] = 'new member value'  # add new member
    params.save('my_params.json')  # save or update params as json file.
    ```
    """

    def __init__(self, json_path):
        """Load json file add dict property"""
        self.json_path = json_path
        with open(json_path) as f:
            params = json.load(f)
            params.update({'json_path': json_path})
            self.__dict__.update(params)

    def save(self, json_path=None):
        """Save params into json file. If no path is provided, save to the original file."""
        if json_path is None:
            json_path = self.json_path
        else:
            self.json_path = json_path
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, new_params: dict, json_path=None):
        """Update current params with a provided dictionary and save to the JSON file."""
        if isinstance(new_params, dict):
            self.__dict__.update(new_params)
            self.save(json_path)  # Save the updated parameters back to the original file
        else:
            raise ValueError("The input to `update` must be a dictionary.")

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_as_json(data_dict, file_path_and_name, write_enable):
    
    # Writing the JSON data to a file   
    # os.makedirs(folder_name, exist_ok=True)
    
    # print(folder_name)
    # data_log_path = f'{folder_name}/{file_name}.json'
    # print(data_log_path)
    if bool(write_enable):
        with open(file_path_and_name, 'w') as wf:
            json.dump(obj=data_dict, fp=wf, indent=4)


def get_structed_log(input_json_path, ignore_end=None):
    '''
    Get the epoch unstructured log into a structured json file for the
    convenient of Params class to load the log.
    '''
    log_dict = {}
    with open(input_json_path, 'r') as f:
        for line in f:
            train_log = json.loads(line.strip())
            if train_log:
                for k, v in train_log.items():
                    if k in log_dict:
                        log_dict[k].append(v)
                    else:
                        log_dict[k] = [v]
    return log_dict

def keep_first_n_line(file_path, lines_keep: int):

    if get_rank() == 0:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # N_lines = len(lines)
        if lines_keep <= len(lines):
            lines = lines[:lines_keep]
        else:
            exit(0)

        with open(file_path, 'w') as f:
            f.writelines(lines)

