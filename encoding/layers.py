import warnings
import struct

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization


class BaseEncoder(layers.Layer):
    def __init__(self, base=2, norm=False, keep_origin=False, precision_dtype=np.float32, trainable=False):
        """ Inner Keras layer to transform values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        keep_origin (bool): Indicate whether keep the original value as an additional channel.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding takes an extra slot.

        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the left for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

        super().__init__(trainable=trainable)
        self.base = base
        self.norm = norm
        self.keep_origin = keep_origin
        self.precision_dtype = precision_dtype
        self.column_width = int(precision_dtype.__name__[-2:])

        self._split_func = lambda x: np.array([[[int(c, base=self.base) for c in row] for row in subarr] for subarr in x], dtype=self.dtype)
        self._padding_func = lambda x: x.rjust(self.column_width - 1, '0')

    def _rebase_func(self, x):
        return np.base_repr(struct.unpack('!i',struct.pack('!f', x))[0], base=self.base)


    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, self.precision_dtype)
        sign_array = np.expand_dims(np.array(inputs < 0), axis=-1)
        x = np.abs(inputs, dtype=self.precision_dtype)
        x = np.vectorize(self._rebase_func)(x)
        x = np.vectorize(self._padding_func)(x)
        x = self._split_func(x)
        
        if self.norm:
            x = x/self.base
        
        x = np.concatenate([sign_array, x], axis=-1, dtype=self.precision_dtype)
        
        if self.keep_origin:
            x = np.concatenate([inputs, x], axis=-1, dtype=self.precision_dtype)

        return tf.convert_to_tensor(x, dtype=self.precision_dtype)


class BaseDecoder(layers.Layer):
    def __init__(self, base=2, precision_dtype=np.float32, trainable=False):
        """ Inner Keras layer to transform values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        keep_origin (bool): Indicate whether keep the original value as an additional channel.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding takes an extra slot.

        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the left for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

        super().__init__(trainable=trainable)
        self.base = base
        self.precision_dtype = precision_dtype
        self.column_width = int(precision_dtype.__name__[-2:])


    def call(self, inputs, training=False):
        inputs = inputs.numpy()

        neg_mask, vals = inputs[:, :, 0].astype(bool), inputs[:, :, 1:].astype(np.int32)
        vals = np.vectorize(lambda x: np.base_repr(x, base=self.base))(vals).astype(str)

        abs_vals = np.apply_along_axis(lambda x: struct.unpack('!f', struct.pack('!i', int("".join(x), base=self.base))), -1, vals).squeeze()

        neg_vals = np.ones(abs_vals.shape)
        neg_vals[neg_mask] = -1

        output = abs_vals*neg_vals

        return tf.convert_to_tensor(output, dtype=self.precision_dtype)

class CustomNormalization(Normalization):
    def __init__(self, axis=-1, dtype=None, mean=None, variance=None, keep_origin=False, **kwargs):
        super().__init__(axis, dtype, mean, variance, **kwargs)
        self.keep_origin = keep_origin

    def call(self, inputs, training=False):
        inputs = tf.expand_dims(inputs, axis=-1)
        normalized_values = super().call(inputs)
        return normalized_values
