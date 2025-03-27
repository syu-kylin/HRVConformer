import torch
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from misc import MetricLogger

@torch.no_grad()
def auc_binary(model, test_daset, device):
    ''' Calculate the AUC value '''    
    model.eval()

    # 0). define return dictionaries
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    # 1). Calculate prediction probability
    label_array, y_score_array = [], []        
    for batch in metric_logger.log_every(test_daset, 50, header):
        data, label = batch
        data = data.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)

        y_score = model(data)

        y_score_array.append(y_score)
        label_array.append(label)

    label_array_np = torch.cat(label_array).cpu().numpy()
    y_score_array_np = torch.cat(y_score_array).cpu().numpy()         # y_score_array may be on the cuda device

    # 3). Calculate auc and get fpr, tpr
    fpr, tpr, _ = roc_curve(label_array_np, y_score_array_np[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    metric_logger.update(roc_auc=roc_auc)
    metric_logger.synchronize_between_processes()
    metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    roc_auc = metric['roc_auc']
    
    return fpr.tolist(), tpr.tolist(), roc_auc