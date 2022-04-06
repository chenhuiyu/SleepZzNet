'''
Date: 2021-01-08 09:56:34
LastEditors: Chenhuiyu
LastEditTime: 2021-03-28 03:12:15
FilePath: \\03-24-SleepZzNet\\data.py
'''
import glob
import os
import re
import shutil

import numpy as np

from plot import plot_label_true
from utils import load_seq_ids


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."

    Args:
        dataset (str): dataset name
        files (list): files dir list
        sid (int): sid

    Returns:
        subject_files: list of subject files names
    """

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"
    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for _, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)

    return subject_files


def load_data(subject_files):
    """[Load data from subject files.]

    Args:
        subject_files (str): subject files(.npz)

    Returns:
        signals, labels: len(signals)=len(labels)=subject files num
    """

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            x = np.squeeze(x)

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)
    signals = np.array(signals, dtype=list)
    labels = np.array(labels, dtype=list)
    return signals, labels


def load_datasets(
    config,
    logger,
    fold_idx=1,
    output_dir='./output/',
    restart=False,
    random_seed=9,
    with_plot=False,
):
    """[load dataset from npz files]

    Args:
        config (str): config includes dataset dir and names
        logger (logger): logger
        # todo:fold_idx for k-fold validation
        fold_idx (int): for k-fold validation. Defaults to 1.
        output_dir (str): path to save training logs,checkpoint,and predicted pictures.
        restart (bool): whether delete exists training logs and start new. Defaults to False.
        random_seed (int): random seed to select subjects file for cross validation. Defaults to 9.
        with_plot (bool): if is true ,plot the true sleep stage changing for every subjects. Defaults to False.

    Returns:
        train_dataset, valid_dataset, test_dataset: each dataset has 3 elements (x, y, length)
    """
    # Create output directory for the specified fold_idx
    if restart:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))

    # Load subject IDs
    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    logger.info("Load generated SIDs from {}".format(fname))
    logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])
    test_sids = fold_pids[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)

    # Further split training set as validation set (10%)
    n_valids = round(len(train_sids) * 0.10)

    # Set random seed to control the randomness
    np.random.seed(random_seed)
    valid_sids = np.random.choice(train_sids, size=n_valids, replace=False)
    train_sids = np.setdiff1d(train_sids, valid_sids)

    logger.info("Train SIDs: ({}) {}".format(len(train_sids), train_sids))
    logger.info("Valid SIDs: ({}) {}".format(len(valid_sids), valid_sids))
    logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

    subject_files = np.hstack(subject_files)

    # if with_plot is true. plot the label's change and saved as png
    if with_plot is True:
        _, labels = load_data(subject_files)
        for i in range(len(labels)):
            plot_label_true(subject_files[i], output_dir, labels[i])
        logger.info("Plot the true sleep stage for all subjects and saved as png")

    # Get corresponding files
    train_dataset = get_split_data(train_sids, dataset=config["dataset"], subject_files=subject_files)
    valid_dataset = get_split_data(valid_sids, dataset=config["dataset"], subject_files=subject_files)
    test_dataset = get_split_data(test_sids, dataset=config["dataset"], subject_files=subject_files)

    return train_dataset, valid_dataset, test_dataset


def get_split_data(sids, dataset, subject_files):
    files = []
    for sid in sids:
        files.append(get_subject_files(
            dataset=dataset,
            files=subject_files,
            sid=sid,
        ))
    files = np.hstack(files)
    dataset = load_data(files)
    # x, y, length = get_mixed_subject_data(dataset)
    return dataset
