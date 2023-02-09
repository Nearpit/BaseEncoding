import tensorflow as tf
import numpy as np
keras = tf.keras
layers = tf.keras.layers
import warnings
import re

class IntegerBaseEncoder(layers.Layer):
    def __init__(self, base=2, norm=True, column_width=32, encode_sign=True, trainable=False):
        """ Keras layer to transform INTEGER values to the chosen base

        Args:
        base (int): Convert number to the base number system. The valid range is 2-36, the default value is 2.
        norm (bool): Indicate whether normalize the rebased numerical values.
        column_width (int): Set up the width of encoded values for the unificaiton. CAVEAT! Sign encoding can take one slot if encoding_sign==True.
        encode_sign (bool): Indicates whether encode a sign of the lead digit into a new column.

        Auxiliary Functions:
        _rebase_func: Rebase numerical values into a new base values
        _padding_func: Pad rebased values with zeros on the right for the column size consistency
        _split_func: Explode strings (e.g. ['10', '00']) into separate integer channels (e.g. [[1, 0], [0, 0]])
        """

        super().__init__(trainable=trainable)
        self.base = base
        self.norm = norm
        self.column_width = column_width
        self.encode_sign = encode_sign

        self._rebase_func = np.vectorize(np.base_repr)
        self._split_func = lambda arr: np.array(
            [[[int(c, base=self.base) for c in row] for row in subarr] for subarr in arr], dtype=np.int32)


    # def _split_func(self, arr):
    #     "Splitting list of rebased values into digits"

    #     print('', end='')
    #     return np.array([[re.findall(r'-?[\d\w]{1}', x) for x in sublist] for sublist in arr], dtype=np.int32)    


    def _padding_func(self, x, column_width=0):
        """ Padding values with zeros to the left in order to ensure that exploding renders to the same size of vectors

        Args:
        x (list(str)) - string rebased values
        column_width - an auxulary variable, in case of Subclass FloatBaseEncoder, to treat mantissa and exponential differently

        """
        using_width = self.column_width
        if column_width:
            using_width = column_width
        vals = np.vectorize(lambda x: x.zfill(using_width-int(self.encode_sign)))(x)
        check_padding = lambda x: np.array([[len(val)for val in rows] for rows in x]).max()
        column_width = check_padding(vals)
        if column_width > (using_width - int(self.encode_sign)):
            raise ValueError(f'column_width is too little. Please, make sure that max_column_width >= {column_width + 1}')
        return vals

    def _muliply_leading(self, arr, neg_values):
        """Multiplying leading digits by -1 in according to the mask of negative values infered before rebasement
        
        Args:
        arr (list(int)) - rebased values
        neg_values (list(bool)) - indicates which values should be negative
        """


        # Get the indices of the first occurance of "1" in each row
        first_occurance = np.argmax(arr != 0, axis=2)

        # Create a mask that is "True" at the first occurance of "1" in each row
        mask = neg_values[..., np.newaxis]*(np.arange(arr.shape[2]) == first_occurance[..., np.newaxis])
        return np.where(mask, -arr, arr)


    def call(self, inputs, column_width=0, *args, **kwargs):
        if not column_width:
            column_width = self.column_width
        neg_values = np.array(inputs < 0)
        x = np.abs(inputs)
        x = self._rebase_func(x, base=self.base)
        x = self._padding_func(x, column_width)
        
        x = self._split_func(x)
        if self.norm:
            x = x / (self.base - 1)

        if self.encode_sign:
            neg_values = np.expand_dims(neg_values.astype(np.int32), axis=-1)
            x = np.concatenate((neg_values, x), axis=-1)
        else:
            x = self._muliply_leading(x, neg_values)
        return x


class FloatBaseEncoder(IntegerBaseEncoder):
    def __init__(self, round_decimal=0, only_integers=False, mantissa_column_width=0, **kwags):
        """ Child class from IntegerBaseEncoder to transform FLOAT values to the chosen base
        
        Args:

        round_decimal (int) - round decimal part to the provided size
        only_integers (bool) - indicate whether tranform np.frexp representation to the genuine computer float representation
        mantissa_column_width (int) - same as column_width but provides an opportunity to control mantissa column size independently 
        **IntegerBaseEncoder Args
        """
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
