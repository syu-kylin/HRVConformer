import os 
import json
import torch
import time as time
import logging

logger = logging.getLogger('project_log')

log_dir = './log/'   
os.makedirs(log_dir, exist_ok=True)
project_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(int(time.time())))

model_name = 'HrvConfermer'
project_name = "4d-HIE-HRV HrvConfermer-weak-label"
code_versn = '4d'

job_name = f"Test"
group_name = 'test' # random search
num_run = 1


optimizer = torch.optim.AdamW
loss_func = torch.nn.CrossEntropyLoss()
optimizer_name = optimizer.__name__

def project_init(job_name, group_name):

    parent_dir = f'{log_dir}/{job_name}/{group_name}/'
    report_file_name = f'{code_versn}-report-{project_time}.txt'
    # model_dir = f'{parent_dir}{code_versn}-model-{run_code}.pth'

    project_config = {

        # project parameters
        'project_name': project_name,
        'model_name': model_name,
        'job_name': job_name,
        'group_name': group_name,  # random search
        'num_run': num_run,
        'code_versn': code_versn,
        'project_time': project_time,
        'parent_dir': parent_dir,
        'outdir': parent_dir,
        'run_name': '',
        'log_fn': report_file_name,
        'run_log_fn': f'{parent_dir}/{code_versn}-run-log-{project_time}.json',
        'log_json_fn': f'{parent_dir}/{code_versn}-log-{project_time}.json',
        'run_config_fn': f'{parent_dir}/{code_versn}-run-config-{project_time}.json',
        'project_config_json_path': f'{parent_dir}/project_init_config.json',
        'seed': 259,
        'notes': '',
        'best_model_path': '',

        'write_enable': True, # save log and figures in local disk
        'wandb_enable': True, # enable wandb
        'save_model': True,

        # Data parameters
        'sfreq': 4,            # resampled frequency
        'window_length': 5,    # 5 mins
        'overlap': 0.8,
        'mean': 0.5349354123958889,
        'std': 0.10205481709220571,
        'min': 0.23238983050852707,
        'max': 0.8566320754717797,
        'min_max_enable': True,

        'set_name_train': 'train',
        'set_name_val': 'validation',
        'set_name_test': 'test',
        'device': 'cuda',
        'num_workers': 2,
        'pin_memory': True,
        'seed_epoch': 92,

        'epochs': 2000,
        'start_epoch': 0,
        'print_freq': 100,
        'batchsize': 1024,
        'accum_iter': 1,
        'resume': False,

        # Optimeizer paramerters:
        'optimizer_name': 'AdamW',
        'loss_func': 'CrossEntropyLoss',
        'learning_rate': 3e-5,
        'beta1': 0.85,
        'epsilon': 1e-8,
        'beta2': 0.998,
        'weight_decay': 0.05,
        'lr_scheduler': 'cosine_warmup',
        'lr_min': 1e-6,
        'lr_T0': 1e-4,
        'lr_Tmult': 2,
        'lr_T0': 60,

        # 
        'init_method' : 'xavier_uniform', 
        'lr_scheduler1' : None, 
        'warmup_epoch' : 200, 

        # model parameters
        'model_name': 'HrvConfermer',
        'input_dim': 1200,
        'patch_size': 80,
        'd_model': 144,
        'num_attention_heads': 6,
        'n_layer': 3,
        'conv_kernel_size': 11,
        'dropout': 0.3,
        'classifier_head': 'fcn',   # 'fcn', 'mlp'
        'fcn_head_kernel_size': 11,
        'mlp_hid_dim': 200,
        'ff_dim_factor': 4,
        'n_class': 2,

        # distribute parameters
        'dist_on_itp': False,
        'distributed': False,
        'dist_eval': False,
        'dist_backend': 'nccl',
        'dist_url': 'env://',
        'world_size': 1,
        'rank': 0,
        'gpu': 0,
    }


    # config_json_path = f'{parent_dir}/project_init_config.json'
    config_json_path = project_config['project_config_json_path']
    # print(data_log_path)
    os.makedirs(parent_dir, exist_ok=True)

    with open(config_json_path, 'w') as wf:
        json.dump(obj=project_config, fp=wf, indent=4)

    message = ''.join([
        '\033[32;1mProject initial config file saved!\033[0m',
        "Project initialized at {}\n\n".format(project_time),
        "Job name: {}\n".format(job_name),
        "Group name: {}\n\n".format(group_name),    
    ])
    logger.info(message)

    return config_json_path

def setup_logger(log_fn):
    logger = logging.getLogger('project_log')
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():  # Avoid adding multiple handlers
        logger.handlers.clear()

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(levelname)s - %(message)s')
    consoleHandler.setFormatter(formatter)              
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(log_fn, mode='w')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    return logger




if __name__ == '__main__':
    
    config_json_path = project_init(job_name, group_name)
    print(config_json_path)

    # log_dir = './log/'
    # job_name = 'Test'
    # group_name = 'test'
    # run_name = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(int(time.time())))
    # parent_dir = f'{log_dir}/{job_name}/{group_name}/{run_name}/'
    # os.makedirs(parent_dir, exist_ok=True)
    # print(run_name)
    # print(parent_dir)