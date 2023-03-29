import optuna
import pickle

import numpy as np 

from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import utilities.constants as constants
import utilities.funcs as funcs
from utilities.dataset import load_toy_dataset

from utilities.custom_layers import BaseEncoder


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
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
    history = model.fit(transformed_x[:train_size], 
                        y[:train_size], 
                        validation_data = (transformed_x[train_size:train_size+valid_size], y[train_size:train_size+valid_size]),
                        epochs=constants.EPOCHS,
#                         epochs = 2,
                        batch_size=constants.BATCH_SIZE,
                        verbose=0,
                        callbacks=callback)
    return np.median(history.history['loss'][-constants.PATIENCE:])


if __name__ == '__main__':
    results = []
    dataset = load_toy_dataset()
    params_id = 0

    train_size = round(constants.TRAIN_SHARE*constants.N_SAMPLES)
    valid_size = round(constants.VALID_SHARE*constants.N_SAMPLES)
    test_size = round(constants.TEST_SHARE*constants.N_SAMPLES)

    for x_name, loader in dataset.items():
        for variable in ['x', 'y_exp', 'y_lin', 'split']:
            exec(f'{variable} = loader[variable]')
        split = np.expand_dims(split, axis=0)[0]
        for y, y_name in zip([y_lin, y_exp], ['lin', 'exp']):
            for transformation_name, transormation_loaders in constants.TRANSFORMATIONS.items():
                for transormation_loader in transormation_loaders:
                    funcs.set_seed(constants.SEED)
                    print(f'x_{x_name}', f'y_{y_name}', transformation_name, transormation_loader['params'])
                    transformed_x = transormation_loader['func'](x)
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=100)
                    trial = study.best_trial

                    print("  Value: {}".format(trial.value))

                    print("  Params: ")
                    for key, value in trial.params.items():
                        print("    {}: {}".format(key, value))
                    print('SCORE')
                    for seed in constants.EXPERIMENT_SEEDS:
                        funcs.set_seed(seed)
                        model = build_model(n_features=transformed_x.shape[-1], **trial.params)

                        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
                        history = model.fit(transformed_x[:train_size], 
                                            y[:train_size], 
                                            validation_data = (transformed_x[train_size:train_size+valid_size], y[train_size:train_size+valid_size]),
                                            epochs=constants.EPOCHS,
                                            batch_size=constants.BATCH_SIZE,
                                            verbose=0,
                                            callbacks=callback)
                        y_hat = model.predict(transformed_x[-test_size:])
                        y_pure = eval(f'funcs.{y_name}_func')(x)
                        score_noised = mse(y_hat, y[-test_size:])
                        score_pure = mse(y_hat, y_pure[-test_size:])
                        print(score_noised, score_pure)
                        results.append({'x':x_name, 
                                        'y':y_name,
                                        'params_id': params_id,
                                        'transformation_name':transformation_name,
                                        'nn_params': trial.params,
                                        'transformation_params': transormation_loader['params'],
                                        'score_noised': score_noised,
                                        'score_pure' : score_pure,
                                        'history':history.history['loss'],
                                        'seed':seed,
                                        'n_samples' : x.shape[0],
                                        'n_features' : x.shape[1],
                                        'epochs' : constants.EPOCHS
                                        })
                    params_id += 1
                    print()
                    with open(f'best_params.pkl', 'wb') as file:
                        pickle.dump(results, file)

