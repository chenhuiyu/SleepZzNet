'''
Date: 2021-03-23 15:08:56
LastEditors: Chenhuiyu
LastEditTime: 2021-03-26 02:18:18
FilePath: \\03-24-SleepZzNet\\plot.py
'''
import os

import matplotlib.pyplot as plt


def plot_sleep_stage(one_subject_label, save_dir, predicted_label=None, subject_id=None):
    """[plot sleep stage]

    Args:
        one_subject_label (list): [description]
        save_dir ([type]): [description]
        predicted_label ([type], optional): [description]. Defaults to None.
        subject_id ([type], optional): [description]. Defaults to None.
    """
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(111)
    plt.xlabel('Time')
    plt.ylabel('Sleep Stage')
    plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
    if predicted_label is None:
        plt.plot(one_subject_label)
    else:
        ax1.set_title("One Night Sleep Stage")
        p1, = plt.plot(one_subject_label)
        p2, = plt.plot(predicted_label, color='orange', alpha=0.5)
        plt.legend([p1, p2], ["true", "predicted"], loc='best')
    ax1.set_title("One Night Sleep Stage for subject {}".format(str(subject_id)))
    plt.savefig("{}.png".format(os.path.join(save_dir, 'subject_' + str(subject_id))))
    plt.close(fig)


def plot_label_true(subject_file, output_dir, labels):
    """[plot the true sleep stage changing for one night]

    Args:
        subject_file (str): one subject file's name
        output_dir (str): photo save directory
        labels (list): one night sleep stage labels
    """
    _, name = os.path.split(subject_file)
    name, _ = os.path.splitext(name)
    save_dir = os.path.join(output_dir, "sleep_stage_plot")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_sleep_stage(labels, save_dir, subject_id=name)


def plot_predicted_stage(y_true, y_pred, output_dir, length, epoch):
    save_dir = os.path.join(output_dir, 'pred', "epoch_" + str(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sum = 0
    for i in range(len(length)):
        label = y_true[sum:sum + length[i]]
        pred = y_pred[sum:sum + length[i]]
        sum += length[i]
        plot_sleep_stage(one_subject_label=label, predicted_label=pred, save_dir=save_dir, subject_id=i)
