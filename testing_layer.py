import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from encoding.layer import IntegerBaseEncoder, FloatBaseEncoder


x = [[1.5, -2, 16, 4],
     [3, 24, -1, 5]]

x = tf.convert_to_tensor(x)
layer = FloatBaseEncoder(base=2, norm=False, max_column_width=4)
dense = tf.keras.layers.Dense(10)

x = layer(x)


print(x)
