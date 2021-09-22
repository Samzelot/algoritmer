from scipy.optimize import approx_fprime

class Problem:
    def __init__(self, f, gradient = None):
        self.f = f
        self.gradient = gradient
        
    def g(self, eps):
        return self.gradient or (lambda x: approx_fprime(x, self.f, eps))
