import numpy as np

class Solver:
    pass

class Newtons_method:

    pass

f = lambda x1,x2: 100(x2-x1**2)**2 + (1-x1)**2

def g(x):
    pass

def G(f,x):
    n = x.shape[0]
    pass
    #returns Hessian


def newtons_method(f,init_g, k = 100):
    for i in range(k):
        s = -np.linalg.inv(G(x))g(x)
        x = x + s
