import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
initializers = tf.keras.initializers


import numpy as np

from encoding.layers import IntegetBaseEncoder, FloatBaseEncoder


x = np.array([[1, 2, -4, 5], 
              [7, -2, 3, 7]])


layer_float = FloatBaseEncoder(column_width=4)
layer_float_only_integers = FloatBaseEncoder(column_width=10, only_integers=True)

layer_integer = IntegetBaseEncoder(column_width=4)


print(layer_float(x), '\n')
# print(layer_float_only_integers(x), '\n')
print(layer_integer(x))