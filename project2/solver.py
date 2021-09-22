import numpy as np
import scipy.linalg as sl
from scipy.optimize import approx_fprime
from abc import abstractmethod, ABC

class Solver(ABC):
    
    @abstractmethod
    def solve(self, f):
        pass

class QuasiNewtonMethod(Solver):
    
    def __init__(self, stopping_criteria, stopping_error):
        self.stopping_error = stopping_error
        self.stopping_criteria = stopping_criteria
        
    def solve(self, problem, guess, iters=1000):
        f = problem.f
        g = problem.gradient or (lambda x: approx_fprime(x, f, 1e-5))
        #TODO: add attribute for epsilon in approx_fprime function

        x = guess
        for i in range(iters):
            H = self.hessian(x, f)

            if sl.det(H) == 0: # Check if determinant is zero
                raise ValueError("Hessian matrix is singular")
                
            s = -np.linalg.inv(H)@g(x) # calculate the next step

            new_x = self.step(x, s)

            if self.stopping_criteria.lower() == "cauchy":
                print(f"np.linalg.norm(x-new_x): {np.linalg.norm}, selfstoppingerror: {self.stopping_error}")
                if np.linalg.norm(x-new_x) < self.stopping_error: #check the cauchy stopping criteria
                    return new_x
            elif self.stopping_criteria.lower() == "residual":
                print(f"np.linalg.norm(g(new_x)): {np.linalg.norm(g(new_x))}, selfstoppingerror: {self.stopping_error}")
                if np.linalg.norm(g(new_x))< self.stopping_error:
                    return new_x
            else:
                raise ValueError("Not recognized stopping criterion")
            x = new_x

        raise RuntimeError("Maximum iterations exceeded")

    @abstractmethod
    def step(self, x, s):
        return
    
    @abstractmethod
    def hessian(self, x, f):
        return


class NewtonsMethod(QuasiNewtonMethod):

    def __init__(self, stopping_criteria, stopping_error, finite_differences_step):
        super().__init__(stopping_criteria, stopping_error)
        self.finite_differences_step = finite_differences_step
        
    def hessian(self, x, f):

        n = x.shape[0] #x is a n dimensional vector (initial guess)

        H = np.zeros((n,n)) # Hessian matrix H to store approximations
        
        e=np.eye(n) #create basis vectors

        h=self.finite_differences_step
        
        
        for i in range(n): # Loop through Hessian matrix entries and approximate entries
            for j in range(n):
                H[i, j] = (f(x+e[i]*h+e[j]*h)-f(x+e[i]*h)-f(x+e[j]*h)+f(x))/h**2
         
        H = 1/2*(H + H.T) # Symmetric step

        return H
    
    def step(self, x, s):
        return x + s

