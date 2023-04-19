import sys
sys.path.append('./')
import glob
import os
import re

import numpy  as np
import scipy.stats as stats

from utilities import constants, funcs
from utilities.funcs import set_seed
import tweedie



def distribution_data_sampler(dist_name, params, n_samples, n_features, **kwargs):
    """Generating dataset based on the given parameters
    
    Args:
    dist_name(str) - name of distribution
    params(dict(str:float)) - parameters of the distribution
    n_samples(int) - number of generating samples
    n_features(int) - number of generating features
    """
    if dist_name == 'tweedie':
        dist = tweedie.tweedie
    elif dist_name == 'multimodal':
        dist = stats.norm
        modal_number = len(params)
    else:
        dist = getattr(stats, dist_name)
    

    if dist_name == 'multimodal':
        data = np.array([[]]).reshape(0, n_features)
        for k, v in params.items():
            vals =  dist.rvs(size=[n_samples//modal_number, n_features], **params[k])
            data = np.vstack((data, vals))
    else:
        data = dist.rvs(size=[n_samples, n_features], **params)
    return np.sort(data, axis=0)

def generate_toy_dataset():
    """ Generating a toy dataset based on predefined distributions.
    """
    set_seed(constants.SEED)
    valid_size = round(constants.VALID_SHARE*constants.N_SAMPLES)
    test_size = round(constants.TEST_SHARE*constants.N_SAMPLES)

    beyond_size = round(constants.LAST_SAMPLES_SHARE*constants.N_SAMPLES)

    index_array = np.arange(constants.N_SAMPLES)

    regular_range, beyond_range = index_array[:-beyond_size], index_array[-beyond_size:]

    test_indices = np.random.choice(regular_range, size = test_size, replace=False)
    regular_range = regular_range[~np.isin(regular_range, test_indices)]
    valid_indices = np.random.choice(regular_range, size = valid_size, replace=False)
    train_indices =  regular_range[~np.isin(regular_range, valid_indices)]
    np.random.shuffle(train_indices)

    split = { 'train' : train_indices,
              'valid' : valid_indices,
              'test' :  { 'beyond' : beyond_range,
                          'within' : test_indices
                         }
              }
    
    noise = stats.norm.rvs(size=(constants.N_SAMPLES, 1), loc=0, scale=1)

    for distribution_name, params_list in constants.DISTRIGBUTIONS.items():
        for param_idx, params in enumerate(params_list):
            if not param_idx:
                param_idx = ''

            x = distribution_data_sampler(distribution_name, params, constants.N_SAMPLES, constants.N_FEATURES)
            y = np.sin(x) + noise

            np.savez_compressed(f'toy_dataset/sin_y/{distribution_name}{param_idx}', 
                                x=x, 
                                y=y,
                                params=params,
                                split=np.array(split))



# def generate_toy_dataset():
#     """ Generating a toy dataset based on predefined distributions.
#     """
#     set_seed(constants.SEED)
#     noise = stats.norm.rvs(size=(constants.N_SAMPLES, 1), loc=constants.NORM_LOC, scale=constants.NORM_SCALE)

#     index_array = np.arange(constants.N_SAMPLES)
#     train_size = round(constants.TRAIN_SHARE*constants.N_SAMPLES)
#     valid_size = round(constants.VALID_SHARE*constants.N_SAMPLES)
#     test_size = round(constants.TEST_SHARE*constants.N_SAMPLES)
#     split = { 'train' : index_array[:train_size],
#               'valid' : index_array[train_size:train_size+valid_size],
#               'test' : index_array[-test_size:]
#               }

#     for distribution_name, params_list in constants.DISTRIGBUTIONS.items():
#         for param_idx, params in enumerate(params_list):
#             if not param_idx:
#                 param_idx = ''
#             x = distribution_data_sampler(distribution_name, params, constants.N_SAMPLES, constants.N_FEATURES)
#             y_lin, y_exp = funcs.lin_func(x) + noise, funcs.exp_func(x) + noise

#             np.savez_compressed(f'toy_dataset/{distribution_name}{param_idx}', 
#                                 x=x, 
#                                 y_exp=y_exp,
#                                 y_lin=y_lin,
#                                 params=params,
#                                 split=np.array(split))

def load_toy_dataset(path='./toy_dataset/*.npz', distribution_subset=list(constants.DISTRIGBUTIONS.keys())):
    """ Loading toy dataset from a directory and returning as a dictionary.
    """
        # DATASET
    dataset = dict()
    for filepath in glob.glob(path):
        filename = os.path.basename(filepath)
        distribution_name = re.findall('(\w*).npz', filename)[0]
        if distribution_name in distribution_subset:
            with np.load(filepath, allow_pickle=True) as file:
                inner_dict = {}
                for element_name in file.files:
                    inner_dict[element_name] = file[element_name]
                dataset[distribution_name] = inner_dict
    return dataset

    
if __name__ == '__main__':
    generate_toy_dataset()
    dataset = load_toy_dataset()
    print('check')