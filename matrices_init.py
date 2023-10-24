import numpy as np

class RandomMatrices:
    """
    Attributes:
        matrices (List): A list to store the generated matrices.
        shape (tuple or integer): Shape same as matrix to be converted

    Methods:
        add(self, amount: int, distribution: str): Appends certain amount of random matrices to the class object.
        __init__(self, distribution: str): Initializes the RandomMatrices object.
        __generate(self, distribution: str) -> List: Inner method, generates the random matrices based on the given distribution.
    """
    def __init__(self, shape, matrices=[]):
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
    
    def add(self, amount: int, **kwargs) -> None: 
        """
        Appends certain amount of random matrices to the class object.

        Args:
            amount (int): The number of matrices.
            **kwargs: Additional keyword arguments for the distribution parameters.

        Keyword Args:
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
        """
        if amount > 0:
            self.matrices = self.__generate(amount, **kwargs)
        else:
            raise ValueError("Amount of matrices should be strictly positive")

    def __generate(self, amount: int, **kwargs) -> list:
        matrices = list(self.matrices)
        distribution = kwargs.get('distribution', 'normal')
        if distribution == 'normal':
            mean = kwargs.get('mean', 0)
            variance = kwargs.get('variance', 1)
            for i in range(amount):
                matrix = np.random.normal(mean, variance, self.shape)
                matrices.append(matrix)
        elif distribution == 'exponential':
            scale = kwargs.get('scale', 1.0)
            for i in range(amount):
                matrix = np.random.exponential(scale, self.shape)
                matrices.append(matrix)
        elif distribution == 'uniform':
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            for i in range(amount):
                matrix = np.random.uniform(low, high, self.shape)
                matrices.append(matrix)
        elif distribution == 'binomial':
            n = kwargs.get('n', 1)
            p = kwargs.get('p', 0.5)
            for i in range(amount):
                matrix = np.random.binomial(n, p, self.shape)
                matrices.append(matrix)
        elif distribution == 'bernoulli':
            p = kwargs.get('p', 0.5)
            for i in range(amount):
                matrix = np.random.binomial(1, p, self.shape)
                matrices.append(matrix)
        elif distribution == 'integer':
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 10)
            for i in range(amount):
                matrix = np.random.randint(low, high, self.shape)
                matrices.append(matrix)
        return matrices
    
    def clear(self):
        self.matrices = np.array([])

b = RandomMatrices((4, 4))
b.add(1, distribution='integer', low=-10, high=10)
print(b.matrices)


