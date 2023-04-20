import pickle
import logging

import numpy as np 
import optuna


from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utilities.constants as constants
import utilities.funcs as funcs
from utilities.dataset import load_toy_dataset
from utilities.custom_layers import BaseEncoder, PreprocessingWrapper


physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level='INFO')

def build_model(depth, width, lr, n_features, l1=0, l2=0):
    model = keras.Sequential()
    regularization = keras.regularizers.L1L2(l1=l1, l2=l2)
    n_features = x.shape[-1]
    new_channel_size = transformed_x.shape[-1]
    model.add(keras.Input(shape=(n_features, new_channel_size)))
    for _ in range(depth - 1):
        model.add(layers.Dense(width, activation=constants.ACTIVATION, kernel_regularizer=regularization))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def objective(trial):
    # Regularization

    regularization = {'l1':trial.suggest_float("l1", constants.DECAY_RANGE[0], constants.DECAY_RANGE[1]), 
                      'l2':trial.suggest_float("l2", constants.DECAY_RANGE[0], constants.DECAY_RANGE[1])}


    n_features = transformed_x.shape[-1]


    depth = trial.suggest_int("depth", constants.NN_DEPTH_RANGE[0], constants.NN_DEPTH_RANGE[-1])
    lr = trial.suggest_float("lr", constants.LR_RANGE[0], constants.LR_RANGE[-1])
    max_width = funcs.get_layer_width(n_features, constants.MAX_NUM_PARAMS + 1, constants.NN_DEPTH_RANGE[-1])
    width = trial.suggest_int(f"width", constants.NN_WIDTH_RANGE[0], max_width) 
    
    model = build_model(depth, width, lr, n_features, **regularization)
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
    history = model.fit(tf.gather_nd(transformed_x ,train_split), 
                        y[train_split], 
                        validation_data = (tf.gather_nd(transformed_x ,valid_split), y[valid_split]),
                        epochs=constants.EPOCHS,
                        batch_size=constants.BATCH_SIZE,
                        verbose=0,
                        callbacks=callback)
    return np.median(history.history['loss'][-constants.PATIENCE:])

pcs = 8 # PRINT_COLUMN_SIZE
if __name__ == '__main__':
    results = []
    dataset = load_toy_dataset('./toy_dataset/sin_y/*.npz')
    params_id = 0

    y_name = 'sin'
    for x_name, loader in dataset.items():
        for variable in ['x', 'y', 'split']:
            exec(f'{variable} = loader[variable]')
        split = np.expand_dims(split, axis=0)[0]
        train_split = split['train'].reshape(-1, 1)
        valid_split = split['valid'].reshape(-1, 1)
        test_within_split = split['test']['within'].reshape(-1, 1)
        test_beyond_split = split['test']['beyond'].reshape(-1, 1)

        for transformation_name, transormation_loaders in constants.TRANSFORMATIONS.items():
            transformation_layer = transormation_loaders['preproc_layer']
            tranformation_params = transormation_loaders['params']
            for params in tranformation_params:
                for keep_origin in [False, True]:
                    for duplication in [1, constants.MAX_NUM_FEATURES]:
                        if (transformation_name == 'numerical_encoding' or transformation_name == 'k_bins_discr') and duplication == constants.MAX_NUM_FEATURES:
                            continue
                        if transformation_name == 'identity' and keep_origin == True:
                            continue 
                        
                        funcs.set_seed(constants.SEED)
                        logging.info(f"{'x':<{pcs}} {'transformation_name':<{int(3*pcs)}} {'params':<{int(7*pcs)}} {'keep_origin':<{int(2*pcs)}}, {'duplication':<{int(2*pcs)}}")
                        logging.info(f'{x_name:<{pcs}} {transformation_name:<{int(3*pcs)}} {str(params):<{int(7*pcs)}} {keep_origin:<{int(2*pcs)}} {duplication:<{int(2*pcs)}}')

                        current_layer = PreprocessingWrapper(transformation_layer(**params), keep_origin=keep_origin, duplicate=duplication)
                        transformed_x = current_layer(x)
                        study = optuna.create_study(direction="minimize")
                        study.optimize(objective, n_trials=constants.OPTUNA_N_TRIALS)

                        trial = study.best_trial

                        logging.info('SCORES')
                        logging.info(f'SPLIT  {"NOISED":<{int(2*pcs)}} {"PURE":<{int(2*pcs)}}')
                        for seed in constants.EXPERIMENT_SEEDS:
                            funcs.set_seed(seed)
                            model = build_model(n_features=transformed_x.shape[-1], **trial.params)

                            callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
                            history = model.fit(tf.gather_nd(transformed_x ,train_split), 
                                                y[train_split], 
                                                validation_data = (tf.gather_nd(transformed_x , valid_split), y[valid_split]),
                                                epochs=constants.EPOCHS,
                                                batch_size=constants.BATCH_SIZE,
                                                verbose=0,
                                                callbacks=callback)
                            predict_within = model.predict(tf.gather_nd(transformed_x ,test_within_split))
                            y_pure_within = np.sin((x[test_within_split]))

                            predict_beyond = model.predict(tf.gather_nd(transformed_x ,test_beyond_split))
                            y_pure_beyond = np.sin((x[test_beyond_split]))

                            score_noised_within = mse(predict_within, y[test_within_split].squeeze(-1))
                            score_noised_beyond = mse(predict_beyond, y[test_beyond_split].squeeze(-1))

                            score_pure_within = mse(predict_within, y_pure_within.squeeze(-1))
                            score_pure_beyond = mse(predict_beyond, y_pure_beyond.squeeze(-1))

                            logging.info(f"WITHIN {score_noised_within:<{int(2*pcs)}.{pcs}f} {score_pure_within:<{int(2*pcs)}.{pcs}f}")
                            logging.info(f'BEYOND {score_noised_beyond:<{int(2*pcs)}.{pcs}f} {score_pure_beyond:<{int(2*pcs)}.{pcs}f}')
                            results.append({'x':x_name, 
                                            'y':y_name,
                                            'params_id': params_id,
                                            'transformation_name':transformation_name,
                                            'nn_params': trial.params,
                                            'transformation_params': params,
                                            'keep_origin':keep_origin,
                                            'duplication': duplication,
                                            'score_noised_within': score_noised_within,
                                            'score_noised_beyond' : score_noised_beyond,
                                            'score_pure_within' : score_pure_within,
                                            'score_pure_beyond' : score_pure_beyond,
                                            'history':history.history,
                                            'seed':seed,
                                            'n_samples' : x.shape[0],
                                            'n_features' : x.shape[1],
                                            'epochs' : constants.EPOCHS
                                            })
                        params_id += 1
                        print()
                        with open(f'only_inputs_results.pkl', 'wb') as file:
                            pickle.dump(results, file)

