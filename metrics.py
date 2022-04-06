'''
Date: 2021-03-19 18:42:28
LastEditors: Chenhuiyu
LastEditTime: 2021-03-26 11:37:31
FilePath: \\03-24-SleepZzNet\\metrics.py
'''

import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def plot_confusion_matrix(
    cm,
    cmap,
    target_names,
    title='Confusion matrix',
    save_path='confuse_matrix.png',
    normalize=True,
):
    """plot confusion matrix function

    Args:
        cm : confusion matrix
        cmap : color type
        target_names (list): sleep stages names
        title (str): title. Defaults to 'Confusion matrix'.
        save_path (str): save path. Defaults to 'confuse_matrix.png'.
        normalize (bool): normalize. Defaults to True.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片,如果想把图存在什么地方就改一下下面的路径,然后dpi设一下分辨率即可。
    plt.savefig(save_path, dpi=180)
    # plt.show()


# 显示混淆矩阵
def get_epoch_metrics(y_true, predictions):
    """calculate every epoch's metrics

    Args:
        y_true ([type]): true label
        predictions ([type]): predicted results

    Returns:
        metrics_list
    """
    MF1_score = f1_score(y_true, predictions, average='macro')
    cohen_kappa = cohen_kappa_score(y_true, predictions)
    per_class_F1 = f1_score(y_true, predictions, average=None)

    metrics_list = [MF1_score, cohen_kappa, per_class_F1]
    return metrics_list


def get_confusion_matrix(y_true, y_pred, save_path, is_test=False):
    """ plot confusion_matrix

    Args:
        y_true ([type]): label
        y_pred ([type]): predicted label
        save_path (str): matrix pictures path
        is_test (bool): select defferent colors for test and train. Defaults to False.
    """
    if is_test is True:
        cmap = plt.get_cmap('Greens')
    else:
        cmap = plt.get_cmap('Blues')
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plt.figure()
    plot_confusion_matrix(conf_mat, target_names=labels, title='Confusion Matrix', save_path=save_path, normalize=False, cmap=cmap)
    plt.close('all')


def print_metrics(logger, epoch, logs, metrics_list):
    """print each epoch metrics to logger

    Args:
        logger
        epoch (int): this epoch num
        logs: training logs of acc
        metrics_list (list): list of MF1_score, cohen_kappa, per_class_F1
    """
    MF1_score, cohen_kappa, per_class_F1 = metrics_list
    logger.info("[epoch {}] | "
                "Acc : {:.3f} | "
                "Val_Acc : {:.3f} | "
                "MF1 : {:.3f} | "
                "kappa : {:.3f} | ".format(
                    epoch,
                    logs["acc"],
                    logs["val_acc"],
                    MF1_score,
                    cohen_kappa,
                ))
    logger.info('---------- | W : {:.3f} | N1 : {:.3f} | N2 : {:.3f} | N3 : {:.3f} | REM : {:.3f} |'.format(
        per_class_F1[0],
        per_class_F1[1],
        per_class_F1[2],
        per_class_F1[3],
        per_class_F1[4],
    ))
