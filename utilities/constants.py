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

# NN 
ACTIVATION = activations.relu
EPOCHS = 25
BATCH_SIZE = 1024


# Encoding
BASES_ARRAY = [2, 3, 4, 8, 16]
EXPERIMENT_SEEDS = range(2)
BOOL_ARRAY = [False, True]
N_BINS_DISCR = 32
N_QUANTILES = 1000

# Generation 
SEED = 123
N_SAMPLES = 10000
N_FEATURES = 1
TEST_SHARE = 0.2
VALID_SHARE = 0.2
TRAIN_SHARE = 1 - TEST_SHARE - VALID_SHARE

# Distributions
NORM_LOC = 5
NORM_SCALE = 1

EXPON_LOC = 1
EXPON_SCALE = 1
EXPON_S = 0.6

DISTRIGBUTIONS = {
    'norm': [{
        'loc':NORM_LOC,
        'scale':NORM_SCALE
        }]
    ,
    'lognorm': [{
        's':EXPON_S,
        'loc':EXPON_LOC,
        'scale':EXPON_SCALE
        }]
}


TRANSFORMATIONS = {'numerical_encoding' : {'preproc_layer': BaseEncoder, 'params':[]},
                  'identity': {'preproc_layer': Layer, 'params':[dict()]},
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
                    'log_transformation' : {'preproc_layer': LogTranformation, 'params':[dict()]}
                 }


for base in BASES_ARRAY:
    for norm in BOOL_ARRAY:
        if base == 2 and norm == True:
            continue
        current_params = {'base':base, 'norm':norm}
        TRANSFORMATIONS['numerical_encoding']['params'].append(current_params)
