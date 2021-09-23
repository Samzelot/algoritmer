from scipy.optimize import approx_fprime

class Problem:
    def __init__(self, f, g = None):
        self.f = f
        self.grad = g
        
    def g(self, eps):
        return self.grad or (lambda x: approx_fprime(x, self.f, eps))
