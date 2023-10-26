from matrices_init import RandomMatrices
from weights import Weights
import numpy as np
import random as rd

class Desummation():
    def __init__(self):
        self.basis : RandomMatrices = None
        self.optimizer : Weights = None

    def error(self):
        if self.optimizer == None:
            raise ValueError("first execute fit command to get an error")
        else:
            return self.optimizer.errors[-1]

    def weights(self):
        if self.optimizer == None:
            raise ValueError("first execute fit command to get an error")
        else:
            return self.optimizer.weights
    
    def matrices(self):
        if self.basis == None:
            raise ValueError("first execute fit command to get an error")
        else:
            return self.basis.matrices
        
    def fit(self, A, amount, **kwargs):
        '''
        Arguments:
            A - matrix to fit
            amount - how many random matrices will be generated
            **kwargs - additional keyword arguments

        Keyword Args:

            For creating random matrices:
                distribution (str): The desired distribution of elements in the matrices. 
                Supported distributions: 'normal', 'exponential', 'uniform', 'binomial', 'bernoulli', 'integer'.

                mean (float): The mean value(s) for the normal distribution. 
                variance (float): The variance value(s) for the normal distribution. 
                scale (float): The scale parameter for the exponential distribution.
                low (float): The lower bound for the uniform distribution.
                high (float): The upper bound for the uniform distribution.
                n (int): The number of trials for the binomial distribution.
                p (float): The probability of success for the binomial and Bernoulli distributions.
                low (int): The lower bound for the integer distribution.
                high (int): The upper bound for the integer distribution.

            For finding weights:
                n_trials (int): Number of optimization trials
                distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
                Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
                Can be added new distance metrics inside class instance if you know what you are doing.
        '''
        A = np.array(A)

        B = RandomMatrices(A.shape)
        self.basis = B
        B.add(amount, **kwargs)

        optimizer = Weights(**kwargs)
        self.optimizer = optimizer
        optimizer.fit(A, B.matrices)
    
    def predict(self, A):
        '''
        Fits to a given matrix A without creating new random matrices
        '''
        return self.optimizer.fit_predict(A, self.basis.matrices)
    
    def fit_predict(self, A, amount, **kwargs):
        '''
        Fits and then predicts to a given matrix A with creating new random matrices
        
        Arguments:
            A - matrix to fit
            amount - how many random matrices will be generated
            **kwargs - additional keyword arguments

        Keyword Args:

            For creating random matrices:
                distribution (str): The desired distribution of elements in the matrices. 
                Supported distributions: 'normal', 'exponential', 'uniform', 'binomial', 'bernoulli', 'integer'.

                mean (float): The mean value(s) for the normal distribution. 
                variance (float): The variance value(s) for the normal distribution. 
                scale (float): The scale parameter for the exponential distribution.
                low (float): The lower bound for the uniform distribution.
                high (float): The upper bound for the uniform distribution.
                n (int): The number of trials for the binomial distribution.
                p (float): The probability of success for the binomial and Bernoulli distributions.
                low (int): The lower bound for the integer distribution.
                high (int): The upper bound for the integer distribution.

            For finding weights:
                n_trials (int): Number of optimization trials
                distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
                Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
                Can be added new distance metrics inside class instance if you know what you are doing.
        '''
        self.fit(A, amount, **kwargs)
        return self.predict(A)
    



