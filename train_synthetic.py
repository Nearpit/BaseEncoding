import pickle
import logging
import argparse

import numpy as np 
import optuna


from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utilities.constants as constants
import utilities.funcs as funcs
from utilities.dataset import load_toy_dataset
from utilities.custom_layers import PreprocessingWrapper


physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level='INFO')

def build_model(depth, width, lr, n_features, l2=0):
    model = keras.Sequential()
    regularization = keras.regularizers.L1L2(l2=l2)
    model.add(keras.Input(shape=(n_features)))
    for _ in range(depth - 1):
        model.add(layers.Dense(width, activation=constants.ACTIVATION, kernel_regularizer=regularization))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


def objective(trial):
    # Regularization

    l2 = trial.suggest_float("l2", constants.DECAY_RANGE[0], constants.DECAY_RANGE[1], log=True)

    n_features = transformed_x.shape[-1]
    depth = trial.suggest_int("depth", constants.NN_DEPTH_RANGE[0], constants.NN_DEPTH_RANGE[1])
    lr = trial.suggest_float("lr", constants.LR_RANGE[0], constants.LR_RANGE[1], log=True)
    width = trial.suggest_int(f"width", constants.NN_WIDTH_RANGE[0], constants.NN_WIDTH_RANGE[1]) 
    
    
    model = build_model(depth, width, lr, n_features, l2)
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE),
                optuna.integration.TFKerasPruningCallback(trial, 'val_mean_squared_error')]
    history = model.fit(tf.gather_nd(transformed_x ,train_split), 
                        y[train_split.ravel()], 
                        validation_data = (tf.gather_nd(transformed_x ,valid_split), y[valid_split.ravel()]),
                        epochs=constants.EPOCHS,
                        batch_size=constants.BATCH_SIZE,
                        verbose=0,
                        callbacks=callback)
    return np.median(history.history['val_mean_squared_error'][-constants.PATIENCE:])

pcs = 8 # PRINT_COLUMN_SIZE
if __name__ == '__main__':
    results = []
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, nargs='?', choices=['norm', 'lognorm', 'uniform', 'loguniform', 'multimodal', 'tweedie'], required=True)
    args = parser.parse_args()
    dataset = load_toy_dataset(f'datasets/synthetic/{args.dataset}.npz')

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
                    for duplication in [False, constants.MAX_NUM_FEATURES]:
                        if (transformation_name == 'numerical_encoding' or transformation_name == 'k_bins_discr') and duplication == constants.MAX_NUM_FEATURES:
                            continue
                        if transformation_name == 'identity' and keep_origin == True:
                            continue 
                        
                        funcs.set_seed(constants.SEED)
                        logging.info(f"{'x':<{pcs}} {'transformation_name':<{int(3*pcs)}} {'params':<{int(7*pcs)}} {'keep_origin':<{int(2*pcs)}}, {'duplication':<{int(2*pcs)}}")
                        logging.info(f'{x_name:<{pcs}} {transformation_name:<{int(3*pcs)}} {str(params):<{int(7*pcs)}} {keep_origin:<{int(2*pcs)}} {duplication:<{int(2*pcs)}}')

                        current_layer = PreprocessingWrapper(transformation_layer(**params), keep_origin=keep_origin, duplicate=duplication)
                        transformed_x = current_layer(x)
                        study = optuna.create_study(direction="minimize", 
                                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=constants.N_STARTUP_TRIALS, 
                                                                                        n_warmup_steps= constants.N_WARMUP_STEPS))
                        study.optimize(objective, n_trials=constants.OPTUNA_N_TRIALS)

                        trial = study.best_trial

                        logging.info('SCORES')
                        logging.info(f'SPLIT  {"NOISED":<{int(2*pcs)}} {"PURE":<{int(2*pcs)}}')
                        for seed in constants.EXPERIMENT_SEEDS:
                            funcs.set_seed(seed)
                            model = build_model(n_features=transformed_x.shape[-1], **trial.params)

                            callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
                            history = model.fit(tf.gather_nd(transformed_x ,train_split), 
                                                y[train_split.ravel()], 
                                                validation_data = (tf.gather_nd(transformed_x , valid_split), y[valid_split]),
                                                epochs=constants.EPOCHS,
                                                batch_size=constants.BATCH_SIZE,
                                                verbose=0,
                                                callbacks=callback)
                            predict_within = model.predict(tf.gather_nd(transformed_x ,test_within_split))
                            y_pure_within = np.sin((x[test_within_split]))

                            predict_beyond = model.predict(tf.gather_nd(transformed_x ,test_beyond_split))
                            y_pure_beyond = np.sin((x[test_beyond_split]))

                            score_noised_within = mse(predict_within, y[test_within_split.ravel()].squeeze(-1))
                            score_noised_beyond = mse(predict_beyond, y[test_beyond_split.ravel()].squeeze(-1))

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
                        with open(f'results/{x_name}.pkl', 'wb') as file:
                            pickle.dump(results, file)

