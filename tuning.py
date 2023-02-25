import optuna
import pickle

import numpy as np 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utilities.constants as constants
import utilities.funcs as funcs
from utilities.dataset import load_toy_dataset

from encoding.layers import FloatBaseEncoder, IntegetBaseEncoder



def build_model(depth, width, lr, n_features):
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


def objective(trial):
    n_features = transformed_x.shape[-1]
    depth = trial.suggest_int("depth", constants.NN_DEPTH_RANGE[0], constants.NN_DEPTH_RANGE[-1])
    lr = trial.suggest_float("lr", constants.LR_RANGE[0], constants.LR_RANGE[-1])
    max_width = funcs.get_layer_width(n_features, constants.MAX_NUM_PARAMS, constants.NN_DEPTH_RANGE[-1])
    width = trial.suggest_int(f"width", constants.NN_WIDTH_RANGE[0], max_width) 
    print(max_width, end='')
    
    model = build_model(depth, width, lr, n_features)
    callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=constants.PATIENCE)]
    history = model.fit(transformed_x, 
                        y, 
                        epochs=constants.EPOCHS,
                        # epochs = 10,
                        batch_size=constants.BATCH_SIZE,
                        verbose=0,
                        callbacks=callback)
    return np.median(history.history['loss'][-constants.PATIENCE:])


if __name__ == '__main__':
    results = []
    experiment_distributions = ['norm', 'expon', 'lognorm']
    dataset = load_toy_dataset(distribution_subset=experiment_distributions)
    for x_dist_name, x in dataset['x'].items():
        for y_dist_name, y in dataset['y'].items():
            for transformation_name, transormation_loaders in constants.TRANSFORMATIONS.items():
                for transormation_loader in transormation_loaders:
                    funcs.set_seed(constants.SEED)
                    print(f'x_{x_dist_name}', f'y_{y_dist_name}', transformation_name, transormation_loader['params'])
                    transformed_x = transormation_loader['func'](x)
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=100)
                    trial = study.best_trial

                    print("  Value: {}".format(trial.value))

                    print("  Params: ")
                    for key, value in trial.params.items():
                        print("    {}: {}".format(key, value))
                    results.append({'x':x_dist_name, 
                                    'y':y_dist_name,
                                    'transformation_name':transformation_name,
                                    'params': transormation_loader['params'],
                                    'params': trial.params
                                    })
                    with open(f'best_params.pkl', 'wb') as file:
                        pickle.dump(results, file)

