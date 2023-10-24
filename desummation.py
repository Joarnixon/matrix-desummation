from matrices_init import RandomMatrices
from weights import Weights
import numpy as np
import random as rd

class Desummation():
    def __init__(self):
        self.basis : RandomMatrices = None
        self.optimizer : Weights = None

    def fit(self, A, amount, **kwargs):
        A = np.array(A)

        B = RandomMatrices(A.shape)
        self.basis = B
        B.add(amount, **kwargs)

        optimizer = Weights(**kwargs)
        self.optimizer = optimizer
        optimizer.fit(A, B.matrices)
    
    def predict(self, A):
        return self.optimizer.fit_predict(A, self.B.matrices)
    



