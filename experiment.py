from utilities.funcs import set_seed
from utilities import constants

import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
initializers = tf.keras.initializers

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from data_generator import generator
from encoding.layers import _BaseEncoder, FloatBaseEncoder
import pickle



for dist_name in ['normal', 'exponential', 'lognormal']:
    gen = generator.DataGenerator(mean=constants.DISTS_LOC, std=constants.DISTS_SCALE, dist=dist_name)
    x, y = gen.generate(n_features=constants.N_FEATURES, n_samples=constants.N_SAMPLES)
    x = np.ravel(x)
    order = np.argsort(x).reshape(-1, 1)
    x, y = x[order], y[order]
    exec(f'x_{dist_name}=x\ny_{dist_name}=y')


standard = lambda x:(x - x.mean())/x.std()

# x_array = ['x_normal', 'x_exponential', 'x_lognormal']
# y_array = ['y_normal', 'y_exponential', 'y_lognormal']

x_array = ['x_normal', 'x_exponential']
y_array = ['y_normal', 'y_exponential']

tranformations = {
                  'intact': [{'params':{'base':10, 'norm':False, 'encode_sign':None, 'only_integers':None}, 'func':lambda x: x}],
                  'standardization': [{'params':{'base':10, 'norm':True, 'encode_sign':None, 'only_integers':None,}, 'func':standard}],
                #   'higher dimensionality': [{'params':{'base':10, 'norm':False, 'encode_sign':None, 'only_integers':None}, 'func':lambda x, n=5: np.power(x, np.arange(1, n))}],
                  'numerical encoding':[]
                 }

i = 0
for base in base_array:
    for norm in norm_array:
        for encode_sign in encode_sign_array:
            for only_integers in only_integers_array:
                params = {'base':base, 'norm':norm, 'encode_sign':encode_sign, 'only_integers': only_integers}
                exec(f"layer_{i} = FloatBaseEncoder(**params)")
                exec(f"func_{i} = lambda x: layer_{i}(x).squeeze(1)")
                curr_dict = {'params':params, 'func': eval(f"func_{i}")}
                i+=1
                tranformations['numerical encoding'].append(curr_dict)

class _MLPBlock(keras.layers.Layer):
    def __init__(self, width, droprate=0, regularization=keras.regularizers.L1L2(0), **kwargs):
        super(_MLPBlock, self).__init__()
        self.dense = layers.Dense(width, 
                                  kernel_initializer=initializers.RandomNormal(seed=seed),
                                  kernel_regularizer=regularization, **kwargs)
#         self.activation = layers.ReLU()
#         self.drop = layers.Dropout(droprate)
#         self.bn = layers.BatchNormalization()
        
    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
#         x = self.activation(x)
#         x = self.drop(x)
#         x = self.bn(x)
        return x

class MLP(keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dim=64, depth=1, **kwargs):
        super().__init__()
        self.depth = depth
        self.hidden_0 = _MLPBlock(hidden_dim, input_shape=(input_dim,), **kwargs)

        for i in range(1, depth):
            setattr(self, f'hidden_{i}', _MLPBlock(hidden_dim))
        self.out = _MLPBlock(output_dim)

    
    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = getattr(self, f'hidden_{i}')(x)
        x = self.out(x)
        return x


epochs = 1500
activation = activation=keras.activations.relu

results = []
width = 128
depth = 2
for seed in seeds:
    set_seed(seed)
    for idx, x_name in enumerate(x_array):
        for idy, y_name in enumerate(y_array):
            cur_x, cur_y = eval(x_name), eval(y_name)
            for transformation_name, transormation_loaders in tranformations.items():
                for transormation_loader in transormation_loaders:
                    print(seed, x_name, y_name, transformation_name, transormation_loader['params'])
                    transformed_x = transormation_loader['func'](cur_x)
                    model = MLP(transformed_x.shape[1], 1, width, depth=depth, activation=activation)
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')


                    history = model.fit(transformed_x, cur_y, epochs=epochs, verbose=0)
                    y_hat = model.predict(transformed_x)
                    score = mse(y_hat, cur_y)
                    results.append({'x':x_name, 
                                    'y':y_name,
                                    'transformation_name':transformation_name,
                                    'params': transormation_loader['params'],
                                    'score':score,
                                    'history':history.history,
                                    'depth': depth-1,
                                    'width': width,
                                    'seed':seed,
                                    'n_samples' : n_samples,
                                    'n_features' : n_features,
                                    'mean' : mean,
                                    'std' : std,
                                    'epochs' : epochs})
                    with open(f'results.pkl', 'wb') as file:
                        pickle.dump(results, file)
