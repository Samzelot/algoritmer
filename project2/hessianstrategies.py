from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as sl

class HessianStrategy(ABC):
    @abstractmethod
    def hessian(self, problem, globals, x):
        return

class FiniteDifferenceHessian(HessianStrategy):

    def hessian(self, problem, globals, x):
        f = problem.f
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


class GoodBroydenHessian(HessianStrategy):

    def hessian(self, problem, globals, x):
        f = problem.f
        g = problem.g(globals.finite_difference_step)

        if hasattr(self, 'x_last'):
            delta= x - self.x_last
            gamma= g(x) - g(self.x_last)
            H =self.H_last+np.outer((delta-self.H_last@gamma)/(delta@self.H_last@gamma),delta)@self.H_last

            self.H_last = H
            self.x_last = x
            return H
        else:
            exact = FiniteDifferenceHessian()
            self.H_last = exact.hessian(problem, globals, x)
            self.x_last = x
            return self.H_last
            
class BadBroydenHessian(HessianStrategy): #TODO fix
    def hessian(self, problem, globals, x):
        f = problem.f
        g = problem.g(globals.finite_difference_step)

        try:
            delta= x - self.x_last
            gamma= g(x) - g(self.x_last)
            H =self.H_last+np.outer((delta-self.H_last@gamma)/(gamma.T@gamma),gamma.T)

            self.H_last = H
            self.x_last = x
            return H
        except AttributeError:
            exact = FiniteDifferenceHessian()
            self.H_last = exact.hessian(problem, globals, x)
            self.x_last = x
            return self.H_last

class SymmetricHessian(HessianStrategy): #TODO fix
    def hessian(self, problem, globals, x):
        f = problem.f
        g = problem.g(globals.finite_difference_step)

        try:
            delta= x - self.x_last
            gamma= g(x) - g(self.x_last)
            u=delta-(self.H_last@gamma)
            a=1/(u.T@gamma)
            H = self.H_last+a*np.outer(u,u.T)

            self.H_last = H
            self.x_last = x
            return H
        except AttributeError:
            exact = FiniteDifferenceHessian()
            self.H_last = exact.hessian(problem, globals, x)
            self.x_last = x
            return self.H_last

class DFP_rank_2_Hessian(HessianStrategy):
    def hessian(self, problem, globals, x):
        f = problem.f
        g = problem.g(globals.finite_difference_step)

        try:
            delta= x - self.x_last
            gamma= g(x) - g(self.x_last)
            H = self.H_last+(np.outer(delta,delta.T)/(delta.T@gamma)) - (self.H_last@np.outer(gamma, gamma.T)@self.H_last)/(gamma.T@self.H_last@gamma)

            self.H_last = H
            self.x_last = x
            return H
        except AttributeError:
            exact = FiniteDifferenceHessian()
            self.H_last = exact.hessian(problem, globals, x)
            self.x_last = x
            return self.H_last

class BFGS_rank_2_Hessian(HessianStrategy):
    def hessian(self, problem, globals, x):
        f = problem.f
        g = problem.g(globals.finite_difference_step)
        try:
            delta= x - self.x_last
            gamma= g(x) - g(self.x_last)
            H = self.H_last + (1+(gamma.T@self.H_last@gamma)/(delta.T@gamma))*((np.outer(delta, delta.T))/(delta.T@gamma))-((np.outer(delta, gamma.T)@self.H_last + np.outer(self.H_last@gamma, delta.T))/(delta.T@gamma))
            
            self.H_last = H
            self.x_last = x
            return H
        except AttributeError:
            exact = FiniteDifferenceHessian()
            self.H_last = exact.hessian(problem, globals,  x)
            self.x_last = x
            return self.H_last
