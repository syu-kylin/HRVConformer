# import os
# import pandas as pd
# import torch
# import time as timer

# from project_init import project_init
# from utils import Params, write_to_file
# from model.ConformerNet import ConfermerNet, init_weights
# from data_loader import read_split_data, data_loader, device_select
# # from train import train_line, model_init
# from postprocessing import postprocessing, train_summary
# from plot_figures import plot_curves


# config_json_path = project_init()
# # print(config_json_path)
# config = Params(config_json_path)
# print(config.model_name)

# logkw = {
#         'file_path': config.file_path,
#         'file_name': config.report_file_name,
#         'write_enable':config.write_enable,
#     }

# ANSeR_num, window_length, seed_epoch = config.ANSeR_num, config.window_length, config.seed_epoch
# train_epochs, val_epochs = read_split_data(ANSeR_num, window_length, seed_epoch, **logkw)
# train_generator, val_generator = data_loader(train_epochs, val_epochs, config)


# learning_rates = [1e-4, 1e-5, 1e-6]

# run_summary_df = pd.DataFrame([])
# for learning_rate in learning_rates:

#     config.learning_rate = learning_rate

#     # Initialize run
#     run_name = timer.strftime('%Y-%m-%d %H-%M-%S',timer.localtime(int(timer.time())))
#     run_path = f"{config.file_path}/{run_name}"
#     os.makedirs(run_path, exist_ok=True)
#     run_config_json = f"{run_path}/{run_name}_config.json"
#     run_log_json = f"{run_path}/{run_name}_log.json"

#     run_dict = {
#         'run_name': run_name,
#         'run_path': run_path,
#         'run_config_json': run_config_json,
#         'run_log_json': run_log_json,
#     }
#     config.dict.update(run_dict)
#     config.save(run_config_json)

#     # Initialize model and optimizer
#     model = model_init(config)

#     betas_adam = (config.beta_1, config.beta_2)
#     optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate, betas=betas_adam, weight_decay=config.weight_deay)
#     loss_func = torch.nn.CrossEntropyLoss()
#     warmup_scheduler = None
#     lr_scheduler1, lr_scheduler2 = None, None

#     # Runing training pipeline
#     fit_parms = {
#             'model': model,
#             'train_daset': train_generator,
#             'val_daset': val_generator,
#             'optimizer': optimizer,
#             'loss_func': loss_func,
#             'warmup_scheduler': warmup_scheduler,
#             'lr_scheduler': [lr_scheduler1, lr_scheduler2],
#             'param': config,
#         }

#     records, my_model = train_line(**fit_parms)
#     postprocessing(my_model, val_epochs, val_generator, loss_func, config)
#     plot_curves(records, config)

#     # training results summary storage
#     run_summary_dict = train_summary(config)
#     run_summary_series = pd.Series(run_summary_dict)
#     run_summary_df = pd.concat([run_summary_df, run_summary_series], axis=1, ignore_index=True)


# # Save training results summary as csv file
# run_summary_df = run_summary_df.T
# fn = f'{config.file_path}/train_summary.csv'
# run_summary_df.to_csv(fn, index=False)