import tensorflow as tf
import numpy as np
keras = tf.keras
layers = tf.keras.layers


class BaseEncoder(layers.Layer):
    def __init__(self, base=2, norm=True, max_column_width=32):
        """ Keras layer to transform numeric values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        max_column_width (int): Set up the width of encoded values for the unificaiton

        
        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the right for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """


        super().__init__()
        self.base = base
        self.norm = norm
        self.max_column_width = max_column_width


        self._rebase_func = np.vectorize(np.base_repr)
        self._padding_func = np.vectorize(lambda x: x.zfill(self.max_column_width))
        self._split_func = lambda arr: np.array([[[int(c) for c in row] for row in subarr] for subarr in arr], dtype=np.int32)

    def call(self, inputs, *args, **kwargs):
        neg_values = np.expand_dims(np.array(inputs < 0, dtype=np.int32), axis=-1)
        x = np.abs(inputs)
        x = self._rebase_func(x, base=self.base)
        x = self._padding_func(x)
        x = self._split_func(x)
        if self.norm:
            x = x / (self.base - 1)
        x = np.concatenate((neg_values, x), axis=-1)
        return x
