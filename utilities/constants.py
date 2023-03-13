from tensorflow.keras import activations
import utilities.funcs as funcs
from encoding.layers import BaseEncoder

# Encoding
BASES_ARRAY = [2, 3, 4, 8, 16, 32, 36]
EXPERIMENT_SEEDS = range(5)
BOOL_ARRAY = [False, True]

# Generation 
SEED = 123
N_SAMPLES = 10000
N_FEATURES = 1
TEST_SHARE = 0.2

# Distributions
DISTS_LOC = 6
DISTS_SCALE = 3
LOGUNIFORM_A = 1
LOGUNIFORM_B = 2


DISTRIGBUTIONS = {
    'norm': [{
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }]
    ,
    'lognorm': [{
        's':DISTS_LOC,
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }],
    'expon': [{
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }],
    'cauchy': [{
            'loc':DISTS_LOC,
            'scale':DISTS_SCALE
            }],
    'laplace': [{
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }],
    'loguniform': [{
        'a': LOGUNIFORM_A,
        'b': LOGUNIFORM_B,
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }],
    'uniform': [{
        'loc':DISTS_LOC,
        'scale':DISTS_SCALE
        }],
}


TRANSFORMATIONS = {
                  'intact': [{'params':{'base':10, 'norm':False, 'keep_origin':None, 'only_integers':None}, 'func':lambda x: x}],
                  
                  'standardization': [{'params':{
                                                    'base':10,
                                                    'norm':True,
                                                    'keep_origin':False,
                                                    'only_integers':None
                                                }, 
                                       'func': lambda x: funcs.apply_normalization(x, keep_origin=False)},

                                      {'params':{
                                                    'base':10,
                                                    'norm':True,
                                                    'keep_origin':True,
                                                    'only_integers':None
                                                }, 
                                       'func': lambda x: funcs.apply_normalization(x, keep_origin=True)}],

                  'numerical encoding':[]
                 }


_i = 0 # needs for uniqueness of functions and layers. Otherwise, override the existed layers.
for base in BASES_ARRAY:
    for norm in BOOL_ARRAY:
        for keep_origin in BOOL_ARRAY:
            for only_integers in BOOL_ARRAY:
                params = {'base':base, 'norm':norm, 'keep_origin':keep_origin, 'only_integers': only_integers}
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
