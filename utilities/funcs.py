import numpy as np
import tensorflow as tf
import random
import os
from encoding.layers import CustomNormalization


def get_num_params(n, width, depth):
    """ Calculating total trainable parameters based on:

    Args:
    n - input feature size
    width - widht of the NN
    depth - depth of the NN
    """
    return ((n+(depth-1)*width+(depth+1))*width)+1


def get_layer_width(n, n_max, depth):
    """ Calculating a width of NN for a smaller input dataset in order to align the number of parameters.

    Args:
    n - input feature size
    n_max - the maximum feature size of all datasets
    depth - depth of the NN
    """
    a = depth - 1
    b = 3 + n
    c = 1 - n_max
    return int(np.round(max((-b + (b ** 2 - 4  * a * c) ** 0.5) / (2*a), (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2*a))))


def set_seed(seed=42):
    """ Setting up the deteministic behaviour"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1) 

def apply_normalization(inputs, keep_origin=False, **kwargs):
    """ A regular keras normalization layer but with the option to store the initial values as an additional channel.
    
    Args:
    keep_origin(bool) - indicating whether to store initial values as an additional channel.
    """
    layer = CustomNormalization(keep_origin=keep_origin, **kwargs)
    layer.adapt(inputs)
    return layer(inputs)