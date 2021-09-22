
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as sl

class HessianStrategy(ABC):
    @abstractmethod
    def hessian(self, problem, x):
        return

class FiniteDifferenceHessian(HessianStrategy):
    def __init__(self, finite_differences_step):
        self.finite_differences_step = finite_differences_step

    def hessian(self, problem, x):
        f = problem.f
        n = x.shape[0] #x is a n dimensional vector (initial guess)
        H = np.zeros((n,n)) # Hessian matrix H to store approximations
        e=np.eye(n) #create basis vectors
        h=self.finite_differences_step
        for i in range(n): # Loop through Hessian matrix entries and approximate entries
            for j in range(n):
                H[i, j] = (f(x+e[i]*h+e[j]*h)-f(x+e[i]*h)-f(x+e[j]*h)+f(x))/h**2
            
        if sl.det(H) == 0: # Check if determinant is zero
            raise ValueError("Hessian matrix is singular")

        H = 1/2*(H + H.T) # Symmetric step
        H_inv = np.linalg.inv(H)
        return H_inv


class GoodBroydenHessian(HessianStrategy):
    def __init__(self, finite_differences_step):
        self.finite_differences_step = finite_differences_step
        
    def hessian(self, problem, x):
        f = problem.f
        g = problem.g(self.finite_differences_step)

        try:
            sigma= x - self.x_last
            gamma= g(x) - g(self.x_last)
            print((sigma-self.H_last@gamma)/(sigma@self.H_last@gamma))
            print(sigma@self.H_last)
            H =self.H_last+np.outer((sigma-self.H_last@gamma)/(sigma@self.H_last@gamma),sigma)@self.H_last

            self.H_last = H
            return H
        except AttributeError:
            exact = FiniteDifferenceHessian(self.finite_differences_step)
            self.H_last = exact.hessian(problem, x)
            self.x_last = x
            return self.H_last
            
#TODO: add "bad Broyden hessian"
#TODO: add "Symmetric Broyden update"
#TODO: add DFP rank-2 update
#TODO: add BFGS rank-2 update            