import tensorflow as tf
import numpy as np
keras = tf.keras
layers = tf.keras.layers
import warnings

class IntegerBaseEncoder(layers.Layer):
    def __init__(self, base=2, norm=True, column_width=32):
        """ Keras layer to transform INTEGER values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding takes one slot.


        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the right for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

        super().__init__()
        self.base = base
        self.norm = norm
        self.column_width = column_width

        self._rebase_func = np.vectorize(np.base_repr)
        self._split_func = lambda arr: np.array(
            [[[int(c, base=self.base) for c in row] for row in subarr] for subarr in arr], dtype=np.int32)

    def _padding_func(self, x, column_width=0):
        using_width = self.column_width
        if column_width:
            using_width = column_width
        vals = np.vectorize(lambda x: x.zfill(using_width-1))(x)
        check_padding = lambda x: np.array([[len(val)for val in rows] for rows in x]).max()
        column_width = check_padding(vals)
        if column_width > (using_width - 1):
            raise ValueError(f'column_width is too little. Please, make sure that max_column_width >= {column_width + 1}')
        return vals

    def call(self, inputs, column_width=0, *args, **kwargs):
        if not column_width:
            column_width = self.column_width
        neg_values = np.expand_dims(
            np.array(inputs < 0, dtype=np.int32), axis=-1)
        x = np.abs(inputs)
        x = self._rebase_func(x, base=self.base)
        x = self._padding_func(x, column_width)
        
        x = self._split_func(x)
        if self.norm:
            x = x / (self.base - 1)
        x = np.concatenate((neg_values, x), axis=-1)
        return x


class FloatBaseEncoder(IntegerBaseEncoder):
    def __init__(self, round_decimal=0, only_integers=False, mantissa_column_width=0, **kwags):
        """ Child class from IntegerBaseEncoder to transform FLOAT values to the chosen base"""
        super().__init__(**kwags)
        self.only_integers = only_integers
        self.round_decimals = round_decimal
        if mantissa_column_width:
            self.mantissa_column_width = mantissa_column_width
        else:
            self.mantissa_column_width = self.column_width

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

    def call(self, inputs, *args, **kwargs):
        mantissa, exponent = self._frexp(inputs)
        exponent = super().call(exponent, column_width=self.column_width, *args, **kwargs)

        if self.only_integers:
            mantissa = self._get_decimals(mantissa)
            mantissa = super().call(mantissa, column_width=self.mantissa_column_width, *args, **kwargs)
        else:
            mantissa = np.expand_dims(mantissa, axis=-1)
            mantissa = np.concatenate(((np.sign(mantissa) < 0).astype(np.float32), np.abs(mantissa)), axis=2)

        x = np.concatenate((mantissa, exponent), axis=-1)

        return x
