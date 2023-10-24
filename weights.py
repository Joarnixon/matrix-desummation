import numpy as np
import scipy.spatial as sp
import optuna
from functools import partial

class Weights:
    """
    Class for decomposing A into a weighted sum of B_matrices.

    Parameters:
    - random_state (int): Random seed for reproducibility.
    - n_trials (int): Number of optimization trials.
    - **kwargs: Additional keyword arguments.
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

    def __init__(self, random_state=21, n_trials=50, **kwargs) -> None:
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.errors = []
        self.supported = ['chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis']
        self.distance = kwargs.get('distance', 'euclidean')
        self.min = None
        self.max = None

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
        B_matrices = np.array(B_matrices)
        if distance not in self.supported:
            raise ValueError("Distance metric is not supported. Append it to class atribute or choose from supported")
        weighted_sum = sum(weight * B for weight, B in zip(self.weights, B_matrices))
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
        self.errors.append(score)
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
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_sum = sum(weight * B for weight, B in zip(self.weights, B_matrices))
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
