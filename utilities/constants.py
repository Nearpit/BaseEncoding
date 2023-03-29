import tensorflow as tf
from tensorflow.keras import activations
import utilities.funcs as funcs
from utilities.custom_layers import BaseEncoder, Duplication

# Encoding
BASES_ARRAY = [2, 3, 4, 8, 16, 32, 36]
EXPERIMENT_SEEDS = range(5)
BOOL_ARRAY = [False, True]

# Generation 
SEED = 123
N_SAMPLES = 10000
N_FEATURES = 1
TEST_SHARE = 0.2
VALID_SHARE = 0.2
TRAIN_SHARE = 1 - TEST_SHARE - VALID_SHARE

# Distributions
NORM_LOC = 0
NORM_SCALE = 0.5

EXPON_LOC = 1
EXPON_SCALE = 1
EXPON_S = 1

DISTRIGBUTIONS = {
    'norm': [{
        'loc':NORM_LOC,
        'scale':NORM_SCALE
        }]
    ,
    'lognorm': [{
        's':EXPON_LOC,
        'loc':EXPON_SCALE,
        'scale':EXPON_S
        }]
}


TRANSFORMATIONS = {
                  'intact': [{'params':{'base':10, 'norm':False, 'keep_origin':False}, 'func':lambda x: x}],
                  
                  'standardization': [{'params':{
                                                    'base':10,
                                                    'norm':True,
                                                    'keep_origin':False,
                                                }, 
                                       'func': lambda x: funcs.apply_normalization(x)}],
                  'duplication' : [{'params':{
                                                    'base':10,
                                                    'norm':False,
                                                    'keep_origin':False,
                                                }, 
                                       'func': lambda x: Duplication(33)(x)},
                                    {
                                    'params':{
                                                    'base':10,
                                                    'norm':True,
                                                    'keep_origin':False,
                                                }, 
                                       'func': lambda x: Duplication(33)(tf.squeeze(funcs.apply_normalization(x), axis=-1))},
                                       ],

                  'numerical encoding':[]
                 }


_i = 0 # needs for uniqueness of functions and layers. Otherwise, override the existed layers.
for base in BASES_ARRAY:
    for norm in BOOL_ARRAY:
        for keep_origin in BOOL_ARRAY:
            params = {'base':base, 'norm':norm, 'keep_origin':keep_origin}
            exec(f"layer_{_i} = BaseEncoder(**params)")
            exec(f"func_{_i} = lambda x: layer_{_i}(x)")
            curr_dict = {'params':params, 'func': eval(f"func_{_i}")}
            TRANSFORMATIONS['numerical encoding'].append(curr_dict)
            _i+=1

# TUNNING 
MAX_NUM_FEATURES = 65
NN_WIDTH_RANGE = [16, 128]
NN_DEPTH_RANGE = [2, 5]
LR_RANGE = [5e-4, 1e-3]
MAX_NUM_PARAMS = funcs.get_num_params(MAX_NUM_FEATURES, NN_WIDTH_RANGE[-1], NN_DEPTH_RANGE[-1])
PATIENCE = 20

# NN 
ACTIVATION = activations.relu
EPOCHS = 1000
BATCH_SIZE = 256
