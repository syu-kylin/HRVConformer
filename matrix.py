import torch
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from misc import MetricLogger
from collections import defaultdict
import logging

logger = logging.getLogger('project_log')



@torch.no_grad()
def auc_binary(model, test_daset, device, epoch_aggregation=False, verbose=False):
    ''' Calculate the AUC value '''    
    model.eval()

    # 0). define return dictionaries
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Collect predicted probabilities and labels by epoch_id
    epoch_probs = defaultdict(list)
    epoch_labels = defaultdict(list)

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    # 1). Calculate prediction probability
    label_array, y_score_array = [], []        
    for batch in metric_logger.log_every(test_daset, 50, header):
        data, label, file_id = batch
        data = data.to(device, non_blocking=True) 
        label = label.to(device, non_blocking=True)

        y_score = model(data)
        probs = torch.softmax(y_score, dim=1)[:, 1]
        preds = torch.argmax(y_score, dim=1)     # predicted label per segment, faster and numerically more stable than using softmax output

        y_score_array.append(y_score)
        label_array.append(label)
        
        if epoch_aggregation:
            # Aggregate probabilities by file_id (epoch_id)
            for prob, label, fid in zip(probs.cpu(), label.cpu(), file_id):
                epoch_probs[fid].append(prob.item())
                epoch_labels[fid].append(label.item())

    # calculate accuracy on the segment level
    if verbose:
        logger.info(f"Evaluate AUC on segment level.")
    label_array_np = torch.cat(label_array).cpu().numpy()
    y_score_array_np = torch.cat(y_score_array).cpu().numpy()         # y_score_array may be on the cuda device

    fpr_seg, tpr_seg, _ = roc_curve(label_array_np, y_score_array_np[:, 1])
    roc_auc_seg = metrics.auc(fpr_seg, tpr_seg)
    metric_logger.update(roc_auc_seg=roc_auc_seg)

    if epoch_aggregation:
        # Convert aggregated probabilities and labels to numpy arrays
        label_epoch_level = []
        prob_epoch_level = []
        if verbose:
            logger.info(f"Evaluate AUC on epoch level.")
        for fid in epoch_probs.keys():
            avg_prob = sum(epoch_probs[fid]) / len(epoch_probs[fid])
            # majority_vote_pred = int(sum(epoch_preds[fid]) >= len(epoch_preds[fid]) / 2)
            label_one_epoch = set(epoch_labels[fid])
            assert len(label_one_epoch) == 1, f"Multiple labels found for file_id {fid}: {label_one_epoch}"
            label_epoch_level.append(list(label_one_epoch)[0])
            prob_epoch_level.append(avg_prob)
        
        assert len(label_epoch_level) == len(prob_epoch_level), "Mismatch in lengths of labels and probabilities"
        fpr_epoch, tpr_epoch, _ = roc_curve(label_epoch_level, prob_epoch_level)
        roc_auc_epoch = metrics.auc(fpr_epoch, tpr_epoch)
        metric_logger.update(roc_auc_epoch=roc_auc_epoch)

    metric_logger.synchronize_between_processes()
    metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    roc_auc_seg = metric['roc_auc_seg']

    if epoch_aggregation:
        roc_auc_epoch = metric['roc_auc_epoch']
        auc_state = {
            'roc_auc_seg': roc_auc_seg,
            'roc_auc_epoch': roc_auc_epoch,
            'fpr_seg': fpr_seg.tolist(),
            'tpr_seg': tpr_seg.tolist(),
            'fpr_epoch': fpr_epoch.tolist(),
            'tpr_epoch': tpr_epoch.tolist(),
        }
        return auc_state
    else:
        return roc_auc_seg