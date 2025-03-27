# ----------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# Deit: https://github.com/facebookresearch/deit
# ----------------------------------------------

import json
import os
import torch
import numpy as np
import random
import time
from collections import deque, defaultdict
from torch import distributed as dist
from pathlib import Path
import logging

logger = logging.getLogger('project_log')

def setup_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        '''This deque function maintained a fixed length queue. When a new value is added, 
        the oldest value is removed. The total and count are updated accordingly. Operations
        like appending and popping from the ends of a list are performed in O(1) time complexity,
        unlike a list O(n).
        '''
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        This function is used to synchronize the count and total between processes.
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        '''
        Example:
        custom_fmt = "Median: {median:.4f} (global_avg: {global_avg:.4f})"
        smoothed_value = SmoothedValue(fmt=custom_fmt)
        smoothed_value.update(10)
        smoothed_value.update(20)
        print(smoothed_value) # Median: 15.0000 (global_avg: 15.0000)
        '''
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        '''This function returns a string representation of the object.
        When the object is printed, this function is called.
        '''
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        '''This function synchronizes the meters between processes, which executes the all_reduce function. 
        This is useful for distributed training.'''
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """This function logs the progress of the training. It prints the header, the iteration number,
        the estimated time of arrival, the meters, the time taken for the iteration, and the data time.
        The iterable is the data loader. The print_freq is the frequency of printing the progress.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        self.add_meter('step_time', SmoothedValue(fmt='{avg:.4f}'))
        self.add_meter('step_data_time', SmoothedValue(fmt='{avg:.4f}'))
        self.add_meter('memory', SmoothedValue(fmt='{avg:.0f}'))
        self.add_meter('epoch_time', SmoothedValue(fmt='{avg:.4f}'))
        # MB = 1024.0 * 1024.0
        GB = 1024.0 * 1024.0 * 1024.0
        for obj in iterable:
            step_data_time = time.time() - end
            yield obj
            step_time = time.time() - end
            if i % print_freq == 0 or i == len(iterable) - 1:
                if torch.cuda.is_available():
                    memory = torch.cuda.max_memory_allocated() / GB
                    self.update(step_time=step_time, step_data_time=step_data_time, memory=memory)
                else:
                    self.update(step_time=step_time, step_data_time=step_data_time)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        self.update(epoch_time=total_time)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    '''
    This function initializes the distributed mode. It checks if the distributed mode is enabled
    and sets the rank, world size, gpu, dist_url, and distributed mode. It also sets the environment
    variables for the rank, world size, and gpu. If the distributed mode is not enabled, it sets the
    distributed mode to False.
    '''
    # check if the distributed mode is enabled
    # whther distributed training is being executed on a specific plateform, 
    # such as Infiniband Transport Protocol (ITP) (HPC Cluster) (Interl TPU)
    # or a custom interconnect for distributed communication.
    
    # Program will be launched with MPI: (`mpirun` or `mpiexec`)
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

    # Program will be launched with `torch.distributed.launch`
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # Program will be launched with `slurm`
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    # Distributed trainng is not enabled. Program will be launched with `python`
    else:
        logger.info('\033[35;1mNot using distributed mode\033[0m')
        # setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    logger.info('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, lr_scheduler, best=False):
    output_dir = Path(args.outdir)
    if loss_scaler is not None:
        if best:
            checkpoint_paths = [output_dir / ('best_model-%s.pth' % args.run_name)]
            args.model_path = checkpoint_paths[0]
        else:
            checkpoint_paths = [output_dir / ('checkpoint.pth')]

        for checkpoint_path in checkpoint_paths:
            # logger.info("Saving checkpoint to %s" % checkpoint_path)
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler=None):
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info("Resume checkpoint from epoch %s." % checkpoint['epoch'])
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("optimizer & lr scheduler & loss scaler loaded!")
            else:
                logger.info("optimizer & loss scaler loaded!")