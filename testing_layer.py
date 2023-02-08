import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from encoding.layers import IntegerBaseEncoder, FloatBaseEncoder


x = [[1, -2, 3, 4],
     [3, 2, -1, 5]]

x = tf.convert_to_tensor(x)
layer = FloatBaseEncoder(base=3, norm=False, column_width=4, mantissa_column_width=11, only_integers=True)

integer_layer = IntegerBaseEncoder(base=2, norm=False, column_width=4, encode_sign=True)
integer_layer_sign = IntegerBaseEncoder(base=2, norm=False, column_width=4, encode_sign=False)

dense = tf.keras.layers.Dense(10)

print(integer_layer(x))
print(integer_layer_sign(x))
