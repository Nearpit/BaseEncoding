import sys
sys.path.append('./')
import glob
import os
import re

import numpy  as np
import scipy.stats as stats

from utilities import constants
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
    return np.sort(data, axis=0)

def generate_toy_dataset():
    """ Generating a toy dataset based on predefined distributions.
    """
    set_seed(constants.SEED)
    for distribution_name, params_list in constants.DISTRIGBUTIONS.items():
        for name, n_features in zip(['x', 'y'], [constants.N_FEATURES, 1]):
            for param_idx, params in enumerate(params_list):
                if not param_idx:
                    param_idx = ''
                data = distribution_data_sampler(distribution_name, params, constants.N_SAMPLES, n_features)
                np.savez_compressed(f'toy_dataset/{distribution_name}{param_idx}_{name}', data=data, params=params)

def load_toy_dataset(path_to_dataset='./toy_dataset/*.npz', distribution_subset=list(constants.DISTRIGBUTIONS.keys())):
    """ Loading toy dataset from a directory and returning as a dictionary.
    """
        # DATASET
    dataset = {
        'x': {},
        'y': {}
    }
    for filepath in glob.glob(path_to_dataset):
        filename = os.path.basename(filepath)
        distribution_name, feat_or_targ = re.findall('(\w*)_(x|y).npz', filename)[0]
        if distribution_name in distribution_subset:
            with np.load(filepath) as file:
                dataset[feat_or_targ][distribution_name] = file['data']
    return dataset

    
if __name__ == '__main__':
    generate_toy_dataset()