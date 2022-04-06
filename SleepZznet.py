'''
Date: 2021-03-22 14:50:20
LastEditors: Chenhuiyu
LastEditTime: 2021-03-28 14:47:07
FilePath: \\03-24-SleepZzNet\\SleepZznet.py
'''
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Lambda, Reshape, Permute, Concatenate, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from resnet import ResNet18
from transformer import build_transformer


# Build the SleepZznet Architecture
def build_SleepZznet(n_epochs=1, frequency=100, classes=5, epoch_length=30, n_filters=1):
    """[Build the SleepZznet architecture]

    Args:
        n_epochs (int): [number of time epochs in this subbatch to be used for classification]. Defaults to 1.
        frequency (int): number of measured values per second. Defaults to 100.
        classes (int): number of classes (sleepphases) to classify epochs into. Defaults to 5.
        epoch_length (int): length of one epoch in seconds. Defaults to 30.

    Returns:
        classification model
    """

    # get the length of every epoch (in seconds and in datapoints)
    epoch_length_points = epoch_length * frequency * n_filters
    # define the shape of all inputs (shape of the subepochs)
    input_shape = (n_epochs, int(epoch_length_points))
    # prepare the list of feature sequences

    # inputs
    inputs = Input(input_shape)
    x = Reshape((n_epochs, int(epoch_length_points), 1))(inputs)
    # TimeDistributed resnet
    x = ResNet18(x, branch=0)
    x = Reshape((n_epochs, -1))(x)
    # transformer
    x = build_transformer(x)
    # fully connected layer + classifier
    outputs = Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    # create model
    model = Model(inputs=inputs, outputs=outputs, name='SleepZznet')

    return model
