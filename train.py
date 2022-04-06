'''
Date: 2021-03-21 15:51:11
LastEditors: Chenhuiyu
LastEditTime: 2021-03-29 01:18:47
FilePath: \\03-28-SleepZzNet\\train.py
'''
import importlib
import logging
import os

import tensorflow as tf

import numpy as np
from callbacks import create_callbacks
from data_load import load_datasets
from data_processing import cal_classes_weight, data_processing
from focal_loss import categorical_focal_loss
from logger import get_logger
from SleepZznet import build_SleepZznet
from utils import print_n_samples_each_class
from data_augment import normalization, combine_N_sleep_epoch, signal_augment, filter_data, combine_overlapped


def train(config_file, output_dir, random_seed):
    """[train model]

    Args:
        config_file (str): the config file includes params for training and dataset
        output_dir (str): path to save training logs,checkpoint,and predicted pictures
        random_seed (int): random seed to select subjects file for cross validation
    """
    # get config from config file
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.train

    # todo:处理这里，表示是否用了单独对某个波段的滤波
    N_Filters = 1

    # Create logger
    tf.get_logger().setLevel(logging.ERROR)
    log_file = os.path.join(output_dir, 'train.log')
    logger = get_logger(log_file, level="info")
    logger.info("-----------------------log created-------------------------")
    logger.info("log file in :{}".format(log_file))

    # load dataset
    train_dataset, valid_dataset, test_dataset = load_datasets(
        config,
        output_dir=output_dir,
        random_seed=random_seed,
        logger=logger,
        with_plot=False,
    )

    # print number of labels per classes and get classes_weight
    logger.info("Train data and labels")
    x_train, y_train, train_length = data_processing(train_dataset, config, logger)
    classes_length = print_n_samples_each_class(y_train)
    logger.info("Val data and labels")
    x_val, y_val, val_length = data_processing(valid_dataset, config, logger)
    print_n_samples_each_class(y_val)
    logger.info("Test data and labels")
    x_test, y_test, test_length = data_processing(test_dataset, config, logger)
    print_n_samples_each_class(y_test)

    CLASSES_WEIGHT, norm_weight = cal_classes_weight(classes_length, logger)

    # x_train, y_train = combine_N_sleep_epoch(data=x_train, labels=y_train, length=train_length, N_SLEEP_EPOCHS=N_SLEEP_EPOCHS, n_fliters=N_Filters)
    # x_val, y_val = combine_N_sleep_epoch(data=x_val, labels=y_val, length=val_length, N_SLEEP_EPOCHS=N_SLEEP_EPOCHS, n_fliters=N_Filters)

    # warp data in Dataset Object
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # the batchsize must be set on the dataset object
    train_data = train_data.batch(config['batch_size'])
    val_data = val_data.batch(config['batch_size'])

    # built SleepZznet
    SleepZznet = build_SleepZznet(
        n_epochs=config['n_sleep_epochs'],
        frequency=config['frequency'],
        classes=config['n_classes'],
        epoch_length=config['epoch_length'],
        n_filters=N_Filters,
    )

    # plot model structure
    tf.keras.utils.plot_model(model=SleepZznet, to_file=os.path.join(output_dir, 'model.png'), show_shapes=True, show_layer_names=False, dpi=300)
    logger.info('Plot model structure to {}'.format(os.path.join(output_dir, 'model.png')))

    with open(os.path.join(output_dir, 'model_summary.log'), 'w') as f:
        SleepZznet.summary(print_fn=lambda x: f.write(x + '\n'))

    # Create callbacks
    callbacks_list = create_callbacks(
        output_dir=output_dir,
        logger=logger,
        SleepZznet=SleepZznet,
        x_val=x_val,
        y_val=y_val,
        BATCHSIZE=config['batch_size'],
        val_length=val_length,
    )

    SleepZznet.compile(
        optimizer='adam',
        # todo: focal loss
        loss=[categorical_focal_loss(alpha=[norm_weight], gamma=2)],
        # todo: weighted Crossentropy loss
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )

    # Model Fit
    # todo: weighted Crossentropy fit
    # SleepZznet.fit(train_data, epochs=config['n_epochs'], validation_data=val_data, callbacks=callbacks_list, class_weight=CLASSES_WEIGHT)
    # todo: focal loss fit
    SleepZznet.fit(
        train_data,
        epochs=config['n_epochs'],
        validation_data=val_data,
        callbacks=callbacks_list,
    )

    # Evaluate
    result = SleepZznet.evaluate(x_test, y_test, verbose=1)
    logger.info('------------Evaluate------------')
    logger.info(dict(zip(SleepZznet.metrics_names, result)))

    # # Save model
    # model_json = SleepZznet.to_json()
    # with open(os.path.join(output_dir, "model.json"), "w") as json_file:
    #     json_file.write(model_json)
    # logger.info("Saved model to disk")
    # SleepZznet.save_weights(os.path.join(output_dir, "SleepZznet.h5"))
    # logger.info("Saved weights to disk")
