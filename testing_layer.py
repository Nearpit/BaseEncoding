import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from encoding.layer import BaseEncoder


x = [[1, -2, 16, 4],
     [3, 255, -256, 5]]

x = tf.convert_to_tensor(x)
layer = BaseEncoder(base=32, max_column_width=4)
output = layer(x)
print(output)
