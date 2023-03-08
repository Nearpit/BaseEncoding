import pickle
import warnings

import numpy as np 
from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utilities.constants as constants
import utilities.funcs as funcs
from utilities.dataset import load_toy_dataset

from encoding.layers import FloatBaseEncoder, IntegetBaseEncoder


import logging
logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.DEBUG)

def build_model(depth, width, lr):
    model = keras.Sequential()
    n_features = x.shape[-1]
    new_channel_size = transformed_x.shape[-1]
    model.add(keras.Input(shape=(n_features, new_channel_size)))
    for _ in range(depth - 1):
        model.add(layers.Dense(width, activation=constants.ACTIVATION))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss)
    return model

if __name__ == '__main__':
    with open('results.pkl', 'rb') as file:
        results = pickle.load(file)
    with open('best_params.pkl', 'rb') as file:
        params = pickle.load(file)
    experiment_distributions = ['norm', 'expon', 'lognorm']
    dataset = load_toy_dataset(distribution_subset=experiment_distributions)
    for x_dist_name, x in dataset['x'].items():
        for y_dist_name, y in dataset['y'].items():
            if (x_dist_name == 'expon')&(y_dist_name=='expon'):
                print(f'SKIP x {x_dist_name} y {y_dist_name}')
                continue
            for transformation_name, transormation_loaders in constants.TRANSFORMATIONS.items():
                for transormation_loader in transormation_loaders:
                    for seed in constants.EXPERIMENT_SEEDS:
                        funcs.set_seed(seed)
                        transformed_x = transormation_loader['func'](x)
                        nn_params = [row['nn_params'] for row in params if (row['transformation_params'] == transormation_loader['params'])and(row['x']==x_dist_name)and(row['y']==y_dist_name)]
                        if not nn_params:
                            logging.warning(f'NO NN PARAMS CONFIGURATION\nx={x_dist_name}|y={y_dist_name}|transformation_name={transformation_name}|params={transormation_loader["params"]}')
                            continue
                        elif len(nn_params) > 1:
                            logging.warning(f'MORE THAN 1 NN PARAMS CONFIGURATION\nx={x_dist_name}|y={y_dist_name}|transformation_name={transformation_name}|params={transormation_loader["params"]}')
                        print(f'seed={seed}, x_{x_dist_name}', f'y_{y_dist_name}', transformation_name, transormation_loader['params'])
                        nn_params = nn_params[0]
                        model = build_model(**nn_params)
                        history = model.fit(transformed_x, 
                            y, 
                            epochs=constants.EPOCHS,
                            batch_size=constants.BATCH_SIZE,
                            verbose=0)
                        y_hat = model.predict(transformed_x)
                        score = mse(y_hat, y)
                        results.append({'x':x_dist_name, 
                                        'y':y_dist_name,
                                        'transformation_name':transformation_name,
                                        'params': transormation_loader['params'],
                                        'score':score,
                                        'history':history.history['loss'],
                                        'depth': nn_params['depth'],
                                        'width': nn_params['width'],
                                        'seed':seed,
                                        'n_samples' : x.shape[0],
                                        'n_features' : x.shape[1],
                                        'mean' : constants.DISTS_LOC,
                                        'std' : constants.DISTS_SCALE,
                                        'epochs' : constants.EPOCHS})
                        with open(f'results.pkl', 'wb') as file:
                            pickle.dump(results, file)
                    