'''
Date: 2021-03-21 14:01:40
LastEditors: Chenhuiyu
LastEditTime: 2021-03-29 01:16:15
FilePath: \\03-28-SleepZzNet\\resnet.py
'''

import tensorflow.keras.layers as layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D, Dropout, MaxPooling1D, ZeroPadding1D, Reshape, TimeDistributed)
from tensorflow.python.keras import regularizers


# Identity Block
def identity_block(input_x, kernel_size, filters, stage, block, branch):
    """
    Implementation of the identity block (or bottleneck block) as described in
    https://arxiv.org/pdf/1512.03385.pdf and implemented in Keras. Altered as described
    in Back (2019)
    # Arguments
        input_x: Input tensor
        kernel_size: default 3, the kernel size of middle conv layer for the main path
        filters: list of integers, the number of filters in the conv layers at the main path
        stage: integer, current stage label, used for generating layer names
        branch: index of the input branch
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor
    """

    # define name basis for the layers in the block
    conv_name_base = 'branch' + branch + '_res' + str(stage) + block + '_branch'
    bn_name_base = 'branch' + branch + '_bn' + str(stage) + block + '_branch'

    # get the filter numbers for the three conv layers
    filters1, filters2, filters3 = filters

    # store the input tensor for later
    x_shortcut = input_x

    # first component (main path)
    x = TimeDistributed(Conv1D(
        filters=filters1,
        kernel_size=1,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        name=conv_name_base + '2a',
    ))(input_x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2a'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    # second component (main path)
    x = TimeDistributed(Conv1D(
        filters=filters2,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b',
    ))(x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2b'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    # third component (main path)
    x = TimeDistributed(Conv1D(
        filters=filters3,
        kernel_size=1,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        name=conv_name_base + '2c',
    ))(x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2c'))(x)

    # final step: addition of main path and recurrent shortcut, activation
    x = layers.add([x, x_shortcut])
    x = TimeDistributed(layers.Activation('relu'))(x)

    return x


# Convolutional Block
def convolutional_block(input_x, kernel_size, filters, stage, block, branch, strides=2):
    """
    Implementation of the convolutional (or standard block) as described in
    https://arxiv.org/pdf/1512.03385.pdf and implemented in Keras. Altered as described
    in Back (2019)
    # Arguments
        input_x: Input tensor
        kernel_size: default 3, the kernel size of middle conv layer for the main path
        filters: list of integers, the number of filters in the conv layers at the main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        branch: index of the input branch
        stride: strides for the first convolutional layer in the block
    # Returns
        Output tensor
    """

    # define name basis for the layers in the block
    conv_name_base = 'branch' + branch + '_res' + str(stage) + block + '_branch'
    bn_name_base = 'branch' + branch + '_bn' + str(stage) + block + '_branch'

    # get the filter numbers for the three conv layers
    filters1, filters2, filters3 = filters

    # store the input tensor for later
    x_shortcut = input_x

    # first component (main path)
    x = TimeDistributed(Conv1D(
        filters=filters1,
        kernel_size=1,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '2a',
        kernel_regularizer=regularizers.l2(0.01),
    ))(input_x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2a'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    # x = Dropout(0.2)(x)

    # second component (main path)
    x = TimeDistributed(
        Conv1D(
            filters=filters2,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b',
            kernel_regularizer=regularizers.l2(0.01),
        ))(x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2b'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    # x = Dropout(0.2)(x)

    # third component (main path)
    x = TimeDistributed(Conv1D(
        filters=filters3,
        kernel_size=1,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        name=conv_name_base + '2c',
        kernel_regularizer=regularizers.l2(0.01),
    ))(x)
    x = TimeDistributed(BatchNormalization(name=bn_name_base + '2c'))(x)
    # x = Dropout(0.2)(x)

    # shortcut path
    x_shortcut = TimeDistributed(Conv1D(
        filters=filters3,
        kernel_size=1,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1',
        kernel_regularizer=regularizers.l2(0.01),
    ))(x_shortcut)
    x_shortcut = TimeDistributed(BatchNormalization(name=bn_name_base + '1'))(x_shortcut)
    # x = Dropout(0.2)(x)

    # final step: addition of main path and recurrent shortcut, activation
    x = layers.add([x, x_shortcut])
    x = TimeDistributed(layers.Activation('relu'))(x)

    return x


def ResNet18(x_input, branch=1):
    """
    Build the ResNet50 architecture as described in https://arxiv.org/pdf/1512.03385.pdf
    and implemented in Keras. Altered as described in Back (2019)
    # Arguments
        input_x: Input tensor
        branch: index of the input branch
    # Returns
        Output tensor
    """

    branch_index = str(branch)

    x = x_input
    # Stage 1
    x = TimeDistributed(Conv1D(
        64,
        5,  # Resnet: 64, 7,
        strides=5,
        padding='valid',
        kernel_initializer='he_normal',
        name='conv1' + branch_index,
        kernel_regularizer=regularizers.l2(0.01)))(x)
    x = TimeDistributed(BatchNormalization(name='bn_conv1' + branch_index))(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(ZeroPadding1D(padding=1, name='pool1_pad' + branch_index))(x)
    x = TimeDistributed(MaxPooling1D(8, strides=2))(x)

    # print(x.shape)

    # Stage 2
    x = convolutional_block(
        input_x=x,
        kernel_size=3,
        filters=[16, 16, 64],  # Resnet: [64, 64, 256]
        stage=2,
        block='a',
        branch=branch_index,
        strides=3,
    )

    # Stage 3
    x = convolutional_block(
        input_x=x,
        kernel_size=3,
        filters=[32, 32, 128],  # Resnet: [128, 128, 512]
        stage=3,
        block='a',
        strides=3,
        branch=branch_index,
    )

    x = identity_block(
        input_x=x,
        kernel_size=3,
        filters=[32, 32, 128],  # Resnet: [128, 128, 512]
        stage=3,
        block='b',
        branch=branch_index,
    )

    # Additional max-pool to reduce the length of the feature sequence by hal
    x = TimeDistributed(MaxPooling1D())(x)

    # Stage 4
    x = convolutional_block(
        input_x=x,
        kernel_size=3,
        filters=[128, 128, 256],  # Resnet: [256, 256, 1024]
        stage=4,
        block='a',
        strides=3,
        branch=branch_index,
    )

    x = identity_block(
        input_x=x,
        kernel_size=3,
        filters=[128, 128, 256],  # Resnet: [256, 256, 1024]
        stage=4,
        block='b',
        branch=branch_index,
    )

    # Stage 5
    x = convolutional_block(
        input_x=x,
        kernel_size=3,
        filters=[256, 256, 512],  # Resnet: [512, 512, 2048]
        stage=5,
        block='a',
        strides=3,
        branch=branch_index,
    )

    x = identity_block(
        input_x=x,
        kernel_size=3,
        filters=[256, 256, 512],  # Resnet: [512, 512, 2048]
        stage=5,
        block='b',
        branch=branch_index,
    )
    x = TimeDistributed(MaxPooling1D(2, strides=2))(x)

    # Dropout layer to prevent overfitting
    x = TimeDistributed(Dropout(0.3))(x)

    return x
