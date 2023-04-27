import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler, KBinsDiscretizer

import utilities.funcs as funcs
from utilities.custom_layers import BaseEncoder, SklearnPreprocessing, CustomNormalization

# TUNNING 
NN_WIDTH_RANGE = [16, 256]
NN_DEPTH_RANGE = [1, 7]
LR_RANGE = [1e-5, 5e-3] 
DECAY_RANGE = [1e-6, 5e-1]

MAX_NUM_FEATURES = 32
MAX_NUM_PARAMS = funcs.get_num_params(MAX_NUM_FEATURES, NN_WIDTH_RANGE[-1], NN_DEPTH_RANGE[-1])
PATIENCE = 20
MIN_DELTA = 1e-4



# NN 
ACTIVATION = activations.relu
EPOCHS = 1000
OPTUNA_N_TRIALS = 100
BATCH_SIZE = 1024


# Encoding
BASES_ARRAY = [2, 3, 4, 6, 8, 12, 16, 24, 32]
EXPERIMENT_SEEDS = range(10)
BOOL_ARRAY = [False, True]
N_BINS_DISCR = 32
N_QUANTILES = 1000

# Generation 
SEED = 221
N_SAMPLES = 10000
N_FEATURES = 1
TEST_SHARE = 0.195
VALID_SHARE = 0.2
TRAIN_SHARE = 0.6
LAST_SAMPLES_SHARE = 1 - TRAIN_SHARE - VALID_SHARE - TEST_SHARE


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
                    'numerical_encoding' : {'preproc_layer': BaseEncoder, 'params':[]}
                 }

LOSSES = {
    "binary": tf.keras.losses.BinaryCrossentropy(),
    "regression" : tf.keras.losses.MeanSquaredError(),
    "classification": tf.keras.losses.CategoricalCrossentropy(),
}


EVAL_METRICS = {
    "binary": [{"name":"binary_crossentropy", "func": tf.keras.metrics.BinaryCrossentropy, "param":{}}, {"name":"Acc", "func":tf.keras.metrics.BinaryAccuracy, "param":{}}, {"name":"AUC", "func":tf.keras.metrics.AUC, "param":{}}],
    "regression" :  [{"name":"mean_squared_error", "func": tf.keras.metrics.MeanSquaredError, "param":{}}],
    "classification": [{"name":"categorical_crossentropy", "func": tf.keras.metrics.CategoricalCrossentropy, "param":{}}, {"name":"Acc", "func":tf.keras.metrics.CategoricalAccuracy, "param":{}}, {"name":"AUC", "func":tf.keras.metrics.AUC, "param":{}}]
}
ACTIVATIONS = {
    "binary" : tf.keras.activations.sigmoid,
    "regression" : tf.keras.activations.linear,
    "classification": tf.keras.activations.softmax,
}

for base in BASES_ARRAY:
    for norm in BOOL_ARRAY:
        if base == 2 and norm == True:
            continue
        current_params = {'base':base, 'norm':norm}
        TRANSFORMATIONS['numerical_encoding']['params'].append(current_params)

#LOGGING
PCS = 8# PRINT_COLUMN_SIZE

# Test set up
# EPOCHS = 10
# OPTUNA_N_TRIALS = 2
# EXPERIMENT_SEEDS = range(1)