import logging 
import pickle

import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utilities.dataset import get_df, preprocess, get_split
from utilities import funcs, constants
from utilities.custom_layers import PreprocessingWrapper

optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level='INFO')
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def build_model(depth, width, lr, n_features, l2=0):
    model = keras.Sequential()
    regularization = keras.regularizers.L1L2(l2=l2)
    n_features = x.shape[-1]
    model.add(keras.Input(shape=(n_features)))
    for _ in range(depth - 1):
        model.add(layers.Dense(width, activation=constants.ACTIVATION, kernel_regularizer=regularization))
    model.add(layers.Dense(y.shape[-1], activation=constants.ACTIVATIONS[configs["objective"]]))
    loss = constants.LOSSES[configs['objective']]
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    metrics_dict = constants.EVAL_METRICS[configs['objective']]
    metrics_array = [x['func'](**x['param']) for x in metrics_dict]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_array)
    return model
def objective(trial):

    l2 = trial.suggest_float("l2", constants.DECAY_RANGE[0], constants.DECAY_RANGE[1], log=True)

    n_features = x.shape[-1]
    depth = trial.suggest_int("depth", constants.NN_DEPTH_RANGE[0], constants.NN_DEPTH_RANGE[1])
    lr = trial.suggest_float("lr", constants.LR_RANGE[0], constants.LR_RANGE[1], log=True)
    width = trial.suggest_int(f"width", constants.NN_WIDTH_RANGE[0], constants.NN_WIDTH_RANGE[1]) 
    
    model = build_model(depth, width, lr, n_features, l2)
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=constants.PATIENCE)]
    history = model.fit(tf.gather_nd(x, train), 
                        y[train.ravel()], 
                        validation_data = (tf.gather_nd(x ,validate), y[validate.ravel()]),
                        epochs=constants.EPOCHS,
                        batch_size=configs['batch_size'],
                        verbose=1,
                        callbacks=callback)
    metric_to_follow = constants.EVAL_METRICS[configs['objective']][0]['name']
    return np.median(history.history[f'val_{metric_to_follow}'][-constants.PATIENCE:])

if __name__ == "__main__":
    funcs.set_seed()
    args = funcs.get_args()
    configs, df = get_df(args.dataset)
    y, x_cat, x_num = preprocess(df=df, df_name=args.dataset, configs=configs)
    train, validate, test = get_split(df)
    params_id = 0
    results = []

    for transformation_name, transormation_loaders in constants.TRANSFORMATIONS.items():
        transformation_layer = transormation_loaders['preproc_layer']
        tranformation_params = transormation_loaders['params']
        for params in tranformation_params:
            for keep_origin in [False, True]:

                # to avoid keep origin to identity since they are the same
                if transformation_name == 'identity' and keep_origin == True:
                    continue 

                funcs.set_seed(constants.SEED)
                logging.info(f"{'transformation_name':<{int(3*constants.PCS)}} {'params':<{int(7*constants.PCS)}} {'keep_origin':<{int(2*constants.PCS)}}")
                logging.info(f'{transformation_name:<{int(3*constants.PCS)}} {str(params):<{int(7*constants.PCS)}} {keep_origin:<{int(2*constants.PCS)}}')
                current_layer = PreprocessingWrapper(transformation_layer(**params), keep_origin=keep_origin)
                x_num_transformed = current_layer(x_num)
                x = tf.concat((x_num_transformed, tf.cast(x_cat, tf.float32)), axis=-1)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=constants.OPTUNA_N_TRIALS)

                trial = study.best_trial

                logging.info('SCORES')
                for seed in constants.EXPERIMENT_SEEDS:
                    funcs.set_seed(seed)
                    model = build_model(n_features=x.shape[-1], **trial.params)

                    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=constants.MIN_DELTA, patience=constants.PATIENCE)]
                    history = model.fit(tf.gather_nd(x ,train), 
                                        y[train.ravel()], 
                                        validation_data = (tf.gather_nd(x , validate), y[validate.ravel()]),
                                        epochs=constants.EPOCHS,
                                        batch_size=configs['batch_size'],
                                        verbose=1,
                                        callbacks=callback)
                    scores =  model.evaluate(tf.gather_nd(x ,test), y[test.ravel()], batch_size=configs['batch_size'], verbose=0)
                    logging.info(f"SEED {seed} {scores}")
                    results.append({'params_id': params_id,
                                    'x' : args.dataset,
                                    'transformation_name':transformation_name,
                                    'nn_params': trial.params,
                                    'transformation_params': params,
                                    'keep_origin':keep_origin,
                                    'metrics': scores,
                                    'history':history.history,
                                    'seed':seed,
                                    'n_samples' : x.shape[0],
                                    'n_features' : x.shape[1],
                                    'epochs' : constants.EPOCHS
                                    })
                params_id += 1
                print()
                with open(f'results/{args.dataset}_results.pkl', 'wb') as file:
                    pickle.dump(results, file)