import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler, KBinsDiscretizer

import utilities.funcs as funcs
from utilities.custom_layers import BaseEncoder, SklearnPreprocessing, CustomNormalization, LogTranformation

# TUNNING 
MAX_NUM_FEATURES = 32
NN_WIDTH_RANGE = [16, 128]
NN_DEPTH_RANGE = [2, 5]
LR_RANGE = [1e-4, 1e-3]
MAX_NUM_PARAMS = funcs.get_num_params(MAX_NUM_FEATURES, NN_WIDTH_RANGE[-1], NN_DEPTH_RANGE[-1])
PATIENCE = 20
DECAY_RANGE = [0, 1]

# NN 
ACTIVATION = activations.relu
EPOCHS = 1000
OPTUNA_N_TRIALS = 100
BATCH_SIZE = 1024


# Encoding
BASES_ARRAY = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32, 36]
EXPERIMENT_SEEDS = range(10)
BOOL_ARRAY = [False, True]
N_BINS_DISCR = 32
N_QUANTILES = 1000

# Generation 
SEED = 42
N_SAMPLES = 10000
N_FEATURES = 1
TEST_SHARE = 0.195
VALID_SHARE = 0.2
TRAIN_SHARE = 1 - TEST_SHARE - VALID_SHARE
LAST_SAMPLES_SHARE = 0.005


DISTRIGBUTIONS = {
    'norm': [{
        'loc':10,
        'scale':5
        }]
    ,
    'lognorm': [{
        's':1,
        'loc':0.3,
        'scale':10e5
        }]
    ,
    'uniform': [{
        'loc':-10,
        'scale':20
        }]
    ,
    'loguniform': [{
        'a': 0.1,
        'b':10,
        'scale':1e6
        }],
    'multimodal': [{
    '1':{'loc' : 3, 'scale': 2},
    '2': {'loc' : 15, 'scale': 4}
    }],
    'tweedie': [{
    'p' : 1.1,
    'mu' : 10, 
    'phi' : 4
    }]
}


TRANSFORMATIONS = {'identity': {'preproc_layer': Layer, 'params':[dict()]},
                  'standardization': {'preproc_layer': CustomNormalization, 'params':[dict()]},
                   'yeo_johnson' : {'preproc_layer': lambda **params: SklearnPreprocessing(PowerTransformer(**params)), 
                                    'params':[{'method':'yeo-johnson', 'standardize':False}, 
                                              {'method':'yeo-johnson', 'standardize':True}]},
                   'quantilation' : {'preproc_layer': lambda **params: SklearnPreprocessing(QuantileTransformer(**params)), 
                                     'params':[{'n_quantiles': N_QUANTILES, 'output_distribution':'uniform'}, 
                                               {'n_quantiles': N_QUANTILES, 'output_distribution':'normal'}]},
                   'min_max_scaler' : {'preproc_layer': lambda **params: SklearnPreprocessing(MinMaxScaler(**params)), 
                                     'params':[dict()]},
                   'k_bins_discr' : {'preproc_layer': lambda **params: SklearnPreprocessing(KBinsDiscretizer(encode='onehot-dense', **params)), 
                                     'params':[{'n_bins': N_BINS_DISCR, 'strategy':'uniform'},
                                               {'n_bins': N_BINS_DISCR, 'strategy':'quantile'},
                                               {'n_bins': N_BINS_DISCR, 'strategy':'kmeans'}]}, 
                    'numerical_encoding' : {'preproc_layer': BaseEncoder, 'params':[]},        
                    # 'log_transformation' : {'preproc_layer': LogTranformation, 'params':[dict()]}
                 }


for base in BASES_ARRAY:
    for norm in BOOL_ARRAY:
        if base == 2 and norm == True:
            continue
        current_params = {'base':base, 'norm':norm}
        TRANSFORMATIONS['numerical_encoding']['params'].append(current_params)

# # Test set up
# EPOCHS = 10
# OPTUNA_N_TRIALS = 2
# EXPERIMENT_SEEDS = range(2)