import numpy as np

class DataGenerator():
    def __init__(self, mean=0.0, std=1.0, noise=False, is_int=False, dist='normal', is_nonlinear=False, nonlinear_func='sigmoid', seed=123) -> None:
    
        """
        Args:
        n_features (int): Number of features in the generated dataset.
        n_samples (int): Number of instances in the generated dataset.
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution (or scale parameter for exponential distribution).
        is_int (bool): Toggle to generate only integers instead of float numbers.
        dist (str): The distribution used for generating data. Choose one from ['normal', 'lognormal', 'uniform', 'exponential']
        is_nonlinear (bool): Choice whether to apply non-linear transformation on the data.
        nonlinear_func (str): The non-linear transformation used to apply to the data.
        seed(int): Set the random seed.

        """

        self.mean = mean
        self.std = std
        self.noise = noise
        self.is_int = is_int
        self.dist = dist
        self.is_nonlinear = is_nonlinear
        self.nonlinear_func = nonlinear_func
        self.seed = seed

    def generate(self, n_features=10, n_samples=1e+4):
        
        """
        Funtion to generate the data from the chosen distribution with provided parameters.

        Args:
        n_features (int): Number of features in the generated dataset.
        n_samples (int): Number of instances in the generated dataset.
        """

        np.random.seed(self.seed)

        if self.dist == 'normal':
            samples = np.random.normal(self.mean, self.std, (n_samples, n_features))
        elif self.dist == 'lognormal':
            norm_samples = np.random.normal(self.mean, self.std, (n_samples, n_features))
            samples = np.exp(norm_samples)
        elif self.dist == 'uniform':
            samples = np.random.uniform(self.mean, self.std, (n_samples, n_features))
        elif self.dist == 'exponential':
            samples = np.random.exponential(self.std, (n_samples, n_features))
        else:
            raise ValueError('Invalid Distribution')

        if self.is_int:
            samples = np.round(samples).astype(int)

        weights = np.random.uniform(0, 1, n_features)

        if self.noise:
            eps = np.random.normal(0, 1)
        else:
            eps = 0

        target = np.dot(samples, weights)

        return (samples, target)


