import tensorflow as tf
import numpy as np
keras = tf.keras
layers = tf.keras.layers
import warnings
import re

class _BaseEncoder(layers.Layer):
    def __init__(self, base=2, norm=True, keep_origin=True, column_width=32, trainable=False):
        """ Inner Keras layer to transform values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        keep_origin (bool): Indicate whether keep the original value as an additional channel.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding takes an extra slot.

        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the right for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

        super().__init__(trainable=trainable)
        self.base = base
        self.norm = norm
        self.keep_origin = keep_origin
        self.column_width = column_width

        self._rebase_func = np.vectorize(np.base_repr)
        self._split_func = lambda arr: np.array(
            [[[int(c, base=self.base) for c in row] for row in subarr] for subarr in arr], dtype=np.int32)
 


    def _padding_func(self, x):
        """ Padding values with zeros to the left in order to ensure that exploding renders to the same size of vectors

        Args:
        x (list(str)) - string rebased values
        """

        vals = np.vectorize(lambda x: x.zfill(self.column_width))(x)
        check_padding = lambda x: np.array([[len(val)for val in rows] for rows in x]).max()
        cur_column_width = check_padding(vals)
        
        if cur_column_width > (self.column_width):
            raise ValueError(f'column_width is too little. Please, make sure that max_column_width >= {cur_column_width}')
        return vals


    def _rebase(self, inputs, *args, **kwargs):

        neg_values = np.array(inputs < 0)
        x = np.abs(inputs)
        x = self._rebase_func(x, base=self.base)
        x = self._padding_func(x)
        
        x = self._split_func(x)
        if self.norm:
            x = x / (self.base - 1)
        neg_values = np.expand_dims(neg_values.astype(np.int32), axis=-1)
        x = np.concatenate((neg_values, x), axis=-1, dtype=np.float32)
        return x




class IntegetBaseEncoder(_BaseEncoder):
    """ Inner Keras layer to transform values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding takes an extra slot.

        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the right for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=False):
        x = self._rebase(inputs=inputs)
        if self.keep_origin:
            inputs = np.expand_dims(inputs, axis=-1)
            x = np.concatenate((inputs, x), axis=-1, dtype=np.float32)
        return x

class FloatBaseEncoder(_BaseEncoder):
    def __init__(self, round_decimal=0, only_integers=False, **kwags):
        """ Child class from IntegerBaseEncoder to transform FLOAT values to the chosen base
        
        Args:

        round_decimal (int) - round decimal part to the provided size
        only_integers (bool) - indicate whether tranform np.frexp representation to the genuine computer float representation
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        column_width (int): Set up the width of encoded values for the unificaiton. Signs encoding take two extra slots.
        CAVEAT. Column_width size of vector will allocated for both - mantissa and exponent, if only_integer=True. 
        """
        super().__init__(**kwags)
        self.only_integers = only_integers
        self.round_decimals = round_decimal

    def _get_decimals(self, values):
        """ Function which splits a float value by dot and return a fractional part """
        signs = np.sign(values)
        initial_shape = values.shape
        values = values.ravel()
        splitted_values = np.array([[int(x) for x in lists[1:]] for lists in np.char.split(values.astype(str), '.')])

        return (splitted_values.reshape(initial_shape)*signs).astype(np.int32)

    def _frexp(self, inputs):
        """
        Modified function from numpy of a float value transformation. 
        """

        mantissa, exponent = np.frexp(inputs)
        exponent = exponent.astype(int)
        if self.round_decimals:
            warnings.warn(f"Be carefull! Mantissa is rounded to the given number of decimals.")
            mantissa = np.around(mantissa, self.round_decimals)
        return mantissa, exponent

    def call(self, inputs, training=False):
        mantissa, exponent = self._frexp(inputs)
        exponent = self._rebase(exponent)

        if self.only_integers:
            mantissa = self._get_decimals(mantissa)
            mantissa = super()._rebase(mantissa)
        else:
            mantissa = np.expand_dims(mantissa, axis=-1)
            mantissa = np.concatenate(((np.sign(mantissa) < 0).astype(np.float32), np.abs(mantissa)), axis=2)

        x = np.concatenate((mantissa, exponent), axis=-1, dtype=np.float32)
        
        if self.keep_origin:
            inputs = np.expand_dims(inputs, axis=-1)
            x = np.concatenate((inputs, x), axis=-1, dtype=np.float32)
        return x
