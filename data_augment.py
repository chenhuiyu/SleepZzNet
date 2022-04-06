'''
Date: 2021-03-24 13:51:44
LastEditors: Chenhuiyu
LastEditTime: 2021-03-28 13:11:04
FilePath: \\03-24-SleepZzNet\\data_augment.py
'''
import numpy as np
import mne


def normalization(data):
    """normalize data:(data-mean)/std
    Args:
        data

    Returns:
        normalized data
    """
    mu = np.mean(data)
    sigma = np.std(data)
    data = (data - mu) / sigma
    return data


def combine_N_sleep_epoch(data, labels, N_SLEEP_EPOCHS, n_fliters=1):
    """combine N adjacent sleep epoch
    Args:
        data
        labels
        N_SLEEP_EPOCHS (int)

    Returns:
        data, labels
    """
    len_after = len(data) // N_SLEEP_EPOCHS
    data = data[:len_after * N_SLEEP_EPOCHS].reshape(len_after, N_SLEEP_EPOCHS, -1)
    # assert data.shape[-1] == 3000 * n_fliters
    labels = labels[:len_after * N_SLEEP_EPOCHS].reshape(len_after, N_SLEEP_EPOCHS)
    assert labels.shape[0] == data.shape[0]

    return data, labels


def combine_overlapped(data, N_SLEEP_EPOCHS, n_fliters=1):
    combined_data = []
    # for every subjects
    for subject_i in range(len(data)):
        subject_data = data[subject_i]
        combined_epochs = []
        # for every epoch in one subject
        for epoch_i in range(len(subject_data)):
            # start epoch repeat itself
            if (epoch_i < N_SLEEP_EPOCHS - 1):
                combined_epoch = np.expand_dims(subject_data[epoch_i], axis=0)
                combined_epoch = np.tile(combined_epoch, [N_SLEEP_EPOCHS, 1])
            # others combined with N early epochs
            else:
                combined_epoch = subject_data[epoch_i - N_SLEEP_EPOCHS + 1]
                combined_epoch = np.expand_dims(combined_epoch, axis=0)
                for offset in range(N_SLEEP_EPOCHS - 1):
                    early_epoch = subject_data[epoch_i - N_SLEEP_EPOCHS + offset + 2]
                    early_epoch = np.expand_dims(early_epoch, axis=0)
                    combined_epoch = np.concatenate([combined_epoch, early_epoch], axis=0)
            combined_epochs.append(combined_epoch)
        combined_epochs = np.array(combined_epochs)
        combined_data.append(combined_epochs)
    return combined_data


def signal_augment(x, y):
    percent = 0.2
    aug_x = np.copy(x)
    aug_y = np.copy(y)
    for i in range(len(aug_x)):
        offset = np.random.uniform(-percent, percent) * aug_x[i].shape[0]
        roll_x = np.roll(aug_x[i], int(offset))
        aug_x[i] = roll_x
    x = np.concatenate((x, aug_x), axis=0)
    y = np.concatenate((y, aug_y), axis=0)
    return x, y


def filter_data(x, FREQUENCY, wave=None):
    if wave == 'theta':
        l_freq = 4
        h_freq = 8
    else:
        l_freq = 0
        h_freq = 40
    filter_x = mne.filter.filter_data(
        x.astype(np.float64),
        FREQUENCY,
        l_freq,
        h_freq,
    )
    return filter_x
