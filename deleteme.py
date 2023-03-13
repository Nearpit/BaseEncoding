from encoding.layers import BaseEncoder, BaseDecoder
import numpy as np 
base = 2
vals = np.array([[0.0, -1, 1., 100, -100], [1.24, 1.000000005, 33.33333, 0.000000000000001, 0.000000000001]])
be_layer = BaseEncoder(base=base)
bd_layer = BaseDecoder(base=base)
output = be_layer(vals)

print(np.isclose(vals, bd_layer(output)))