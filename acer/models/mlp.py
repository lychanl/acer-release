from typing import List, Callable, Tuple

import tensorflow as tf

from utils import kaiming_initializer, normc_initializer


def build_mlp_network(layers_sizes: Tuple[int] = (256, 256), activation: str = 'tanh',
                      initializer: Callable = None) \
        -> List[tf.keras.Model]:
    """Builds Multilayer Perceptron neural network

    Args:
        layers_sizes: sizes of hidden layers
        activation: activation function name
        initializer: callable to the weight initializer function

    Returns:
        created network
    """
    if activation is None:
        activation = 'tanh'
    if initializer is None:
        if activation.lower() == 'relu':
            initializer = kaiming_initializer()
            bias_initializer = 'zeros'
        else:
            initializer = normc_initializer()
            bias_initializer = 'zeros'
    layers = [
        tf.keras.layers.Dense(
            layer_size,
            activation=activation if i < len(layers_sizes) - 1 else None,
            kernel_initializer=initializer,
            bias_initializer = bias_initializer
        ) for i, layer_size in enumerate(layers_sizes)
    ]

    return layers
