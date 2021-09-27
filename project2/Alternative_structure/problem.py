import numpy as np

class Problem:
    def __init__(self, function, gradient = None, h = 1e-6):
        self.function = function
        if gradient == None:
            self.gradient = lambda x: self.grad(x, h)  # x has to be array type, not int, even for one dimensional problem 
        else:
            self.gradient = gradient

    def grad(self, x, h):   # Numerical approximation of gradient
        n = len(x)
        gradient = np.zeros(n)
        e = np.eye(n)
        for i in range(n):
            gradient[i] = (self.function(x + h*e[i]) - self.function(x)) / h
        return gradient







