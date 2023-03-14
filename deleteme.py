from encoding.layers import BaseEncoder, BaseDecoder, Duplication
import numpy as np 
base = 2
vals = np.array([0.0, -1, 1., 100, -100, 1.24, 1.000000005, 33.33333, 0.000000000000001, 0.000000000001]).reshape(-1, 1)
be_layer = BaseEncoder(base=base)
bd_layer = BaseDecoder(base=base)
output = be_layer(vals)
dup_layer = Duplication(width=4)

reversed_values = bd_layer(output)
print(np.isclose(vals, reversed_values))
dup_result = dup_layer(vals)
print(dup_result)