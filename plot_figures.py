import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import os
import numpy as np

from utils import save_as_json, Params

plt.style.use(['science', 'no-latex'])     # ploy with science style
mpl.use('agg')                             # only write to file (avoid warning of GUI outside of main thread)

def plot_curves(config):
    """ 
    This function will generate the train and validation curves.
    Save them into each run folder.
    Args:
        config: (class of Params) the the dictionary of run config.
    """
    param_log = Params(config.log_json_fn)

    # ################# Plot learning rate and validation AUC of the best model #################
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8), constrained_layout=True)

    axs[0,0].plot(param_log.lr)
    axs[0,0].set_title(f'learning rate Schedule')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].grid()

    labels = ['validation', 'test']
    acc_before = [param_log.val_acc_bm, param_log.test_acc_bm]
    acc_after = [param_log.val_acc_post, param_log.test_acc_post]

    max_acc = max(acc_before + acc_after)
    # print(acc_before, acc_after)
    x = np.arange(len(acc_before))
    width = 0.3

    rects1 = axs[0,1].bar(x - width/2, acc_before, width) 
    rects2 = axs[0,1].bar(x + width/2, acc_after, width)

    # Add values above bars
    def add_annotations(rects):
        for rect in rects:
            height = rect.get_height()
            axs[0,1].text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    add_annotations(rects1)
    add_annotations(rects2)

    axs[0,1].set_ylabel('accuracy')
    axs[0,1].set_title('val and test accuracy before & after postprocessing')
    axs[0,1].set_xticks(x,labels)
    axs[0,1].set_ylim(0.5,max_acc+0.1)
    axs[0,1].legend(['before postprocessing','after postprocessing'])

    axs[1,0].plot(param_log.fpr_val, param_log.tpr_val, label=f'ROC: {param_log.val_auc_bm:.4f}', lw=2)
    axs[1,0].set_title('AUC-ROC on Validation set')
    axs[1,0].set_xlabel('False Positive Rate')
    axs[1,0].set_ylabel('True Positive Rate')
    axs[1,0].legend(loc='lower right', bbox_to_anchor=(0.95, 0.01))

    axs[1,1].plot(param_log.fpr_test, param_log.tpr_test, label=f'ROC: {param_log.test_auc_bm:.4f}', lw=2)
    axs[1,1].set_title('AUC-ROC on test set')
    axs[1,1].set_xlabel('False Positive Rate')
    axs[1,1].set_ylabel('True Positive Rate')
    axs[1,1].legend(loc='lower right', bbox_to_anchor=(0.95, 0.01))
    
    if config.write_enable:
        folder_name = config.outdir
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(f'{folder_name}/{config.code_versn}-lr AUC-{config.run_name}.svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    # ################ Plot the training curves # ##################
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,12), constrained_layout=True)

    text1 = ''.join([
        'lr:{:.2e}\nwd:{}\n'.format(config.learning_rate, config.weight_decay),
        'beta1:{:.4f}\nbeta2:{:.4f}\n'.format(config.beta1, config.beta2),
        'dropout:{}\nn layer:{}'.format(config.dropout, config.n_layer),
    ])
    
    axs[0,0].plot(param_log.train_loss, '.-')
    axs[0,0].set_title('Training loss of each epoch')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Loss')
    axs[0,0].grid()
    axs[0,0].text(0.7, 0.7, text1, fontsize=12, transform=axs[0,0].transAxes)
    
    axs[0,1].plot(param_log.val_loss, '-')
    axs[0,1].set_title('Validation loss of each epoch')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('Loss')
    axs[0,1].grid()
    axs[0,1].text(0.7, 0.7, text1, fontsize=12, transform=axs[0,1].transAxes)
    
    axs[1,0].plot(param_log.train_acc)
    axs[1,0].set_title('Training accuracy of each epoch')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('Accuracy')
    axs[1,0].grid()
    
    axs[1,1].plot(param_log.val_acc)
    axs[1,1].set_title('Validation accuracy of each epoch')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Accuracy')
    axs[1,1].grid()
    
    axs[2,0].plot(param_log.train_auc)
    axs[2,0].set_title('Training AUC of each epoch')
    axs[2,0].set_xlabel('Epoch')
    axs[2,0].set_ylabel('AUC')
    axs[2,0].grid()
    
    axs[2,1].plot(param_log.val_auc, label='val auc')
    axs[2,1].plot(param_log.moving_val_auc, label='moving avg val auc')
    axs[2,1].set_title('Validation AUC of each epoch')
    axs[2,1].set_xlabel('Epoch')
    axs[2,1].set_ylabel('AUC')
    axs[2,1].grid()
    axs[2,1].legend(loc='lower right', bbox_to_anchor=(0.95, 0.01))
    
    if config.write_enable:
        folder_name = config.outdir
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(f'{folder_name}/{config.code_versn}-Train Curve-{config.run_name}.svg', bbox_inches='tight', pad_inches=0.05)

    plt.close()
