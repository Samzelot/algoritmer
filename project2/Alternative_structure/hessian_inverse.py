import numpy as np

def finite_differences(problem, x): 
    f = problem.function
    n = x.shape[0] #x is a n dimensional vector (initial guess)
    H = np.zeros((n,n)) # Hessian matrix H to store approximations
    e=np.eye(n) #create basis vectors
    h=globals.finite_difference_step
    for i in range(n): # Loop through Hessian matrix entries and approximate entries
        for j in range(n):
            H[i, j] = (f(x+e[i]*h+e[j]*h)-f(x+e[i]*h)-f(x+e[j]*h)+f(x))/h**2
            
    if sl.det(H) == 0: # Check if determinant is zero
        raise ValueError("Hessian matrix is singular")

    H = 1/2*(H + H.T) # Symmetric step
    H_inv = np.linalg.inv(H)
    return H_inv


def good_broyden(problem, H_last, x, x_last):
    f = problem.function
    g = problem.gradient

    if H_last == None:
        return finite_differences(problem)
    else:
        delta= x - x_last
        gamma= g(x) - g(x_last)
        H = H_last + np.outer((delta-H_last@gamma)/(delta@H_last@gamma),delta)@H_last

        self.H_last = H
        return H

def bad_broyden():
    pass

def symmetric_broyden():
    pass

def DFP_rank2():
    pass

def BFGD_rank2():
    pass