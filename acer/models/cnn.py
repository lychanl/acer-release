from typing import List, Callable

import tensorflow as tf

from utils import normc_initializer


def build_cnn_network(filters: tuple = (32, 64, 64), kernels: tuple = (8, 4, 3),
                      strides: tuple = ((4, 4), (2, 2), (1, 1)), activation: str = 'relu',
                      initializer: Callable = normc_initializer) \
        -> List[tf.keras.Model]:
    """Builds predefined CNN neural network

    Args:
        filters: tuple with filters to be used
        kernels: tuple with kernels to be used
        strides: tuple with strides to be used
        activation: activation function to be used in each layer
        initializer: callable to the weight initializer function

    Returns:
        created network
    """
    assert len(filters) == len(kernels) == len(strides), "Layers specifications must have the same lengths"
    expand_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
    cast_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
    layers = [expand_layer, cast_layer] + [
        tf.keras.layers.Conv2D(
            cnn_filter,
            kernel,
            strides=stride,
            activation=activation,
            kernel_initializer=initializer(),
            padding="same"
        ) for cnn_filter, kernel, stride in zip(filters, kernels, strides)
    ]

    layers.append(tf.keras.layers.Flatten())
    return layers
