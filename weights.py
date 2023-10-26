import numpy as np
import scipy.spatial as sp
import optuna
import random as rd
from functools import partial

class Weights:
    """
    Class for decomposing A into a weighted sum of B_matrices.

    Parameters:
    - **kwargs: Additional keyword arguments.
        - random_state(int): Random seed for reproducibility.
    """

    def __init__(self, **kwargs) -> None:
        self.random_state = kwargs.get('random_state', rd.randint(0, 100000))
        self.error = 0
        self.weights = None
    
    def loss(self, basis, target, distance='fro'):
        return np.linalg.norm(target - np.tensordot(self.weights, basis, axes=1), ord=distance)

    def fit(self, A, B_matrices):
        """
        Fit the weights to decompose A into a weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        """
        target = np.array(A)
        shape = target.shape
        basis = np.array(B_matrices)
        k = len(basis)

        C = np.zeros((shape[0] * shape[1], k))

        for i in range(k):
            C[:, i] = basis[i].flatten()
        p = np.array(target).flatten()

        # Use the least squares method to solve the system
        x, residuals, rank, singular_values = np.linalg.lstsq(C, p, rcond=None)

        # The solution vector x contains the values of w1, w2, w3, ..., wk
        w = x[:k]
        self.weights = w
        self.error = self.loss(basis=basis, target=target)

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