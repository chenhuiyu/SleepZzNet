'''
Date: 2021-01-08 09:56:34
LastEditors: Chenhuiyu
LastEditTime: 2021-03-28 03:22:37
FilePath: \\03-24-SleepZzNet\\config\\sleepedfx.py
'''
params = {
    # Train
    "n_epochs": 2000,
    "batch_size": 256,
    "epoch_length": 30,
    # "learning_rate": 1e-4,
    "learning_rate": 1e-3,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Signal
    "n_sleep_epochs": 1,
    "n_channels": 1,
    "frequency": 100,
    "filter_theta_wave": True,

    # Early-stopping
    "no_improve_epochs": 50,

    # Model
    "model": "model-mod-8",
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "learning_rate_decay": 0.003,
    "decay_steps": 50000,
    "decay_rate": 0.9,

    # Dataset
    "dataset": "sleepedfx",
    "data_dir": "E:\\EEG\\EEG-Sleep\\sleepedf\\sleep-cassette\\eeg_fpz_cz"",
    "n_folds": 10,
    "n_subjects": 78,

    # Data Augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": True,
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 256,
})

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})
