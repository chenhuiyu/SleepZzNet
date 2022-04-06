'''
Date: 2021-03-28 03:15:47
LastEditors: Chenhuiyu
LastEditTime: 2021-03-28 21:24:15
FilePath: \\03-28-SleepZzNet\\data_processing.py
'''
import numpy as np
from data_augment import normalization, combine_N_sleep_epoch, signal_augment, filter_data, combine_overlapped


def data_processing(dataset, config, logger):
    x, y = dataset
    # x_theta = x.copy()
    # for i in range(len(x)):
    #     x[i] = filter_data(x[i], config['frequency'])
    # if config['filter_theta_wave'] is True:
    #     for i in range(len(x)):
    #         x_theta[i] = filter_data(x[i], config['frequency'], 'theta')
    #         x[i] = np.concatenate([x[i], x_theta[i]], axis=1)

    # x = combine_overlapped(data=x, N_SLEEP_EPOCHS=config['n_sleep_epochs'])

    x, y, length = get_mixed_subject_data(x, y)
    x, y = combine_N_sleep_epoch(
        data=x,
        labels=y,
        N_SLEEP_EPOCHS=config['n_sleep_epochs'],
    )
    x = normalization(x)
    assert (x.shape[0] == y.shape[0])
    logger.info('x shape is {}'.format(x.shape))
    logger.info('y shape is {}'.format(y.shape))
    return x, y, length


def cal_classes_weight(classes_length, logger):
    """calculate weights for each class based on the num of classes

    Args:
        classes_length (dict): dict of the num of each sleep stage classes
        logger (loggger]): logger

    Returns:
        # todo: only saved ones when decied which loss function to use
        CLASSES_WEIGHT, list(norm_weight): CLASSES_WEIGHT(for weighted cross entropy), normalized weight(for focal loss)
    """
    CLASSES_WEIGHT = {}

    classes_num = list(classes_length.values())
    norm_weight = np.zeros(shape=(len(classes_num)))
    for class_i in range(len(classes_num)):
        CLASSES_WEIGHT[class_i] = (round(max(classes_num) / classes_num[class_i]))
        norm_weight[class_i] = (CLASSES_WEIGHT[class_i])
    norm_weight = norm_weight / (np.sum(norm_weight))
    logger.info("CLASSES_WEIGHT:{}".format(CLASSES_WEIGHT))
    logger.info("CLASSES_NORM_WEIGHT:{}".format(norm_weight))
    return CLASSES_WEIGHT, list(norm_weight)


def get_mixed_subject_data(signals, labels):
    length = []
    for i in range(len(labels)):
        length.append(len(labels[i]))
    train_x = np.concatenate(signals, axis=0)
    train_y = np.concatenate(labels, axis=0)
    # train_x = np.expand_dims(train_x, axis=-1)
    train_y = np.array(train_y, dtype=np.float32)
    train_y = train_y.reshape(-1, 1)
    return train_x, train_y, length
