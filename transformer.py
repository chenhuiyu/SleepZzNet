'''
Date: 2021-03-21 02:12:02
LastEditors: Chenhuiyu
LastEditTime: 2021-03-29 15:47:29
FilePath: \\03-28-SleepZzNet\\transformer.py
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.core import Dropout


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.01)),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return super().get_config()


class PositionEmbedding(layers.Layer):
    """Position Embedding Layer
    """
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        pos_emb_output = self.pos_emb(positions)
        return pos_emb_output + x

    def get_config(self):
        # config = super().get_config().copy()
        return super().get_config()


def build_transformer(inputs):
    """build_transformer

    Args:
        inputs: inputs of transformer(output of resnet)

    Returns:
        transformer outputs: outputs has the same shape with input
    """
    # hyperparameter
    # Number of attention heads
    num_heads = 4
    # Hidden layer size in feed forward network inside transformer
    ff_dim = 512
    _, maxlen, embed_dim = inputs.shape

    # Position Embedding
    embedding_layer = PositionEmbedding(maxlen=maxlen, embed_dim=embed_dim)
    x = embedding_layer(inputs)

    # transformer block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # dropouts
    x = layers.Dropout(0.3)(x)
    return x
