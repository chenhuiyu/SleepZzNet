'''
Date: 2021-03-22 15:21:27
LastEditors: Chenhuiyu
LastEditTime: 2021-03-26 11:21:07
FilePath: \\03-24-SleepZzNet\\callbacks.py
'''

import os

import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from metrics import get_confusion_matrix, get_epoch_metrics, print_metrics
from plot import plot_predicted_stage


def create_callbacks(output_dir, logger, SleepZznet, x_val, y_val, BATCHSIZE, val_length):
    # 创建检查点的路径和检查点管理器（manager）。这将用于在每 n个周期（epochs）保存检查点。
    # 保存checkpoints
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'checkpoint', 'weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    # Tensorboard
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, 'Tensorboard'),
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=False,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')

    # save the confuse matrix
    confuse_matrix_dir = os.path.join(output_dir, 'confuse_matrix')
    if not os.path.exists(confuse_matrix_dir):
        os.mkdir(confuse_matrix_dir)

    class metrics_Callback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_path = os.path.join(confuse_matrix_dir, 'val_{:2d}-{:.2f}.png'.format(epoch, logs['val_acc']))
            predictions = np.argmax(SleepZznet.predict(x_val, batch_size=BATCHSIZE), axis=-1)
            assert len(predictions) == len(y_val)
            y_ture = y_val.flatten()
            y_pred = predictions.flatten()
            metrics_list = get_epoch_metrics(y_true=y_ture, predictions=y_pred)
            print_metrics(logger, epoch, logs, metrics_list)
            get_confusion_matrix(y_true=y_ture, y_pred=y_pred, save_path=save_path, is_test=False)
            # if epoch % 10 == 0:
            plot_predicted_stage(
                y_true=y_ture,
                y_pred=y_pred,
                output_dir=output_dir,
                length=val_length,
                epoch=epoch,
            )

    callbacks_list = [checkpointer, tensorboard, metrics_Callback()]
    return callbacks_list
