'''
Date: 2021-03-22 19:00:32
LastEditors: Chenhuiyu
LastEditTime: 2021-03-26 17:54:32
FilePath: \\03-25-SleepZzNet\\focal_loss.py
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Binary focal loss."""

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.0):
    """Softmax focal loss for sparse integer labels.

    Args:
        alpha: class weights, shape (num_classes,) or list-like.
        gamma: focusing parameter.
    """
    alpha = np.array(alpha, dtype=np.float32).reshape(-1)

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=num_classes, dtype=tf.float32)

        epsilon = K.epsilon()
        y_pred_clipped = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Class weights: broadcast to prediction shape
        alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_t = tf.reshape(alpha_t, [1, -1])
        # Fallback if alpha length mismatches num_classes
        alpha_t = tf.cond(
            tf.equal(tf.shape(alpha_t)[1], num_classes),
            lambda: alpha_t,
            lambda: tf.ones([1, num_classes], dtype=tf.float32),
        )

        cross_entropy = -y_true * K.log(y_pred_clipped)
        loss = alpha_t * K.pow(1 - y_pred_clipped, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed
