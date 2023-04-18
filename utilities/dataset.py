import sys
sys.path.append('./')
import glob
import os
import re

import numpy  as np
import scipy.stats as stats

from utilities import constants, funcs
from utilities.funcs import set_seed


def distribution_data_sampler(dist_name, params, n_samples, n_features, **kwargs):
    """Generating dataset based on the given parameters
    
    Args:
    dist_name(str) - name of distribution
    params(dict(str:float)) - parameters of the distribution
    n_samples(int) - number of generating samples
    n_features(int) - number of generating features
    """
        
    dist = getattr(stats, dist_name)
    data = dist.rvs(size=[n_samples, n_features], **params)
    return data


def generate_toy_dataset():
    """ Generating a toy dataset based on predefined distributions.
    """
    set_seed(constants.SEED)
    index_array = np.arange(constants.N_SAMPLES)
    train_size = round(constants.TRAIN_SHARE*constants.N_SAMPLES)
    valid_size = round(constants.VALID_SHARE*constants.N_SAMPLES)
    test_size = round(constants.TEST_SHARE*constants.N_SAMPLES)
    split = { 'train' : index_array[:train_size],
              'valid' : index_array[train_size:train_size+valid_size],
              'test' : index_array[-test_size:]
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

def load_toy_dataset(path_to_dataset='./toy_dataset/*.npz', distribution_subset=list(constants.DISTRIGBUTIONS.keys())):
    """ Loading toy dataset from a directory and returning as a dictionary.
    """
        # DATASET
    dataset = dict()
    for filepath in glob.glob(path_to_dataset):
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