import numpy as np
import scipy.spatial as sp
import optuna
import random as rd
from functools import partial

class Weights:
    """
    Class for decomposing A into a weighted sum of B_matrices.

    Parameters:
    - random_state (int): Random seed for reproducibility.
    - **kwargs: Additional keyword arguments.
        - n_trials (int): Number of optimization trials
        - distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
        - Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
        - Can be added new distance metrics inside class instance if you know what you are doing.

    Attributes:
    - study: Optuna study object.
    - weights (list): List of weights for the B_matrices.
    - random_state (int): Random seed for reproducibility.
    - n_trials (int): Number of optimization trials.
    - errors (list): List of errors during optimization.
    - distance (str): Distance metric to use for calculating the error.
    - supported (list): List of supported distance metrics.
    - min: Minimum value in A.
    - max: Maximum value in A.
    """

    def __init__(self, random_state=rd.randint(0, 10000), **kwargs) -> None:
        self.random_state = random_state
        self.error = []
        self.supported_scipy = ['chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis']
        self.supported_numpy = ['fro', 'nuc', np.inf, -np.inf, 1, 2, -1, -2]
        self.distance = kwargs.get('distance', 'fro')
        self.n_trials = kwargs.get('n_trials', 50)
        self.distance_library = None
        self.min = None
        self.max = None
        self.study = None
        self.weights = None

    def __distance(self, A, B_matrices) -> float:
        """
        Calculate the distance between A and the weighted sum of B_matrices.

        Parameters:
        - A (ndarray): True values.
        - B_matrices (list): List of predicted matrices.

        Returns:
        - distance (float): Distance between A and the weighted sum of B_matrices.
        """
        distance = self.distance
        if distance in self.supported_numpy:
            self.distance_library = 'numpy'
        elif distance in self.supported_scipy:
            self.distance_library = 'scipy'
        else:
            raise ValueError("Distance metric is not supported. Append it to class atribute or choose from Numpy: 'fro', 'nuc', 'inf', '-inf', '0', '1', '2', '-1', '-2', Scipy: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'")
            return False
        
        B_matrices = np.array(B_matrices)
        weighted_sum = sum(weight * B for weight, B in zip(self.weights, B_matrices))

        if self.distance_library == 'numpy':
            return np.linalg.norm(A - weighted_sum, ord=distance)
        else:
            return np.sum(sp.distance.cdist(A, weighted_sum, distance))

    def __objective(self, trial, A, B_matrices):
        """
        Objective function for optimization.

        Parameters:
        - trial: Optuna trial object.
        - A (ndarray): True values.
        - B_matrices (list): List of predicted matrices.

        Returns:
        - score (float): Error score.
        """
        self.weights = [trial.suggest_float(f"weight{n}", -self.max, self.max) for n in range(len(B_matrices))]
        score = self.__distance(A=A, B_matrices=B_matrices)
        self.error.append(score)
        return score

    def fit(self, A, B_matrices):
        """
        Fit the weights to decompose A into a weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        """
        self.min = np.min(A)
        self.max = np.max(A)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self.__objective, A=A, B_matrices=B_matrices)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(B_matrices))]

    def predict(self, B_matrices):
        """
        Parameters:
        - B_matrices (list): List of matrices which will be used to decompose.

        Returns:
        - weighted_sum (ndarray): Weighted sum of B_matrices.
        """
        B_matrices = np.array(B_matrices)
        assert self.weights is not None, 'Weights error, must be fitted before predict'
        weighted_sum = np.tensordot(self.weights, B_matrices, axes=1)
        return weighted_sum

    def fit_predict(self, A, B_matrices):
        """
        Fit the weights and returns the weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        """
        self.fit(A, B_matrices)
        return self.predict(B_matrices)
