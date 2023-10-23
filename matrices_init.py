import numpy as np

class RandomMatrices:
    """
    A class for generating random matrices based on a given distribution.

    Args:
        distribution (str): The desired distribution of elements in the matrices.

    Attributes:
        matrices (List): A list to store the generated matrices.

    Methods:
        __init__(self, distribution: str): Initializes the RandomMatrices object.
        __generate(self, distribution: str) -> List: Generates the random matrices based on the given distribution.
    """

    def __init__(self, shape, matrices=[], mean=0, variance=1):
        self.matrices = np.array(matrices)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise ValueError("Invalid shape provided. Please pass a tuple (rows, columns) or an integer for square matrix")
        if matrices:
            if not all(matrix.shape == self.shape for matrix in matrices):
                raise ValueError("Invalid matrices provided. Please pass with equal shapes.")
            self.matrices = matrices
    
    def add(self, amount : int, distribution : str, mean=0, variance=1) -> None: 
        if isinstance(mean, (list, tuple)) and isinstance(variance, (list, tuple)):
            if len(mean) != len(variance):
                raise ValueError("Length of mean and variance should be the same")
            self.mean = mean
            self.variance = variance
        else:
            self.mean = [mean] * amount
            self.variance = [variance] * amount
        if amount > 0:
            self.__generate(distribution, amount)
        else:
            raise ValueError("Amount of matrices should be strictly positive")

    def __generate(self, distribution: str, amount: int) -> list:
        if distribution == 'normal':
            for i in range(amount):
                matrix = np.random.normal(self.mean[i], self.variance[i], self.shape)
                self.matrices.append(matrix)
        return self.matrices
    
    def generate_matrices(self, amount: int, **kwargs) -> list:
        self.shape = kwargs.get('shape', (3, 3))
        if self.distribution == 'normal':
            self.mean = kwargs.get('mean', 0)
            self.variance = kwargs.get('variance', 1)
        elif self.distribution == 'exponential':
            self.scale = kwargs.get('scale', 1.0)
        elif self.distribution == 'uniform':
            self.low = kwargs.get('low', 0.0)
            self.high = kwargs.get('high', 1.0)
        elif self.distribution == 'binomial':
            self.n = kwargs.get('n', 1)
            self.p = kwargs.get('p', 0.5)
        elif self.distribution == 'bernoulli':
            self.p = kwargs.get('p', 0.5)
        elif self.distribution == 'integer':
            self.low = kwargs.get('low', 0)
            self.high = kwargs.get('high', 10)
        return self.__generate(amount)

a = RandomMatrices('normal')
