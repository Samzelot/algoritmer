from abc import ABC, abstractmethod
from hessianstrategies import FiniteDifferenceHessian
from solver import QuasiNewtonMethod
import numpy as np
from problem import Problem
from scipy.optimize import approx_fprime

class StepStrategy(ABC):
    @abstractmethod
    def step(self, problem, x, s):
        return

class DefaultStep(StepStrategy):
    def step(self, problem, x, s):
        return x + s

class ExactLineStep(StepStrategy):
    def __init__(self, line_solver, finite_differences_step):
        self.line_solver = line_solver
        self.finite_differences_step = finite_differences_step

    def step(self, problem, x, s):
        f_line_x = lambda alpha: x + alpha*s
        line_problem = Problem(
            lambda alpha: problem.f(f_line_x(alpha)),
            #TODO this is not working! why? creates singular hessian after a while
            #lambda alpha: np.array([problem.g(self.finite_differences_step)(f_line_x(alpha))@s/np.linalg.norm(s)]),
        )
        alpha = self.line_solver.solve(line_problem, np.array([1]))
        return f_line_x(alpha)

class InexactLineStep(StepStrategy):
    def __init__(self, finite_differences_step):
        self.finite_differences_step = finite_differences_step
        self.sigma = np.linspace(0.2, 1/2-0.01,50)

    def step(self, problem, x,s):
        phi = lambda alpha: problem.f(x + alpha*s) # phi
        #phi_prime = lambda alpha: problem.g(self.finite_differences_step)(x + alpha*s)@s/np.linalg.norm(s) # phi prime calculation
        phi_prime = lambda alpha : approx_fprime(x+ alpha*s, problem.f, self.finite_differences_step)@(s/np.linalg.norm(s)) 
        alpha_minus = 3 # random alpha minus start
        
        for sigma_val in self.sigma: # loop through sigma values
            while phi(alpha_minus) <= phi(0)+sigma_val*alpha_minus*phi_prime(0): # Armijo rule: alpha_minus
                alpha_minus = alpha_minus/2
                #print(f"phi_prime: {phi_prime(0)}")
                #print(f" {phi(alpha_minus)} <= {phi(0)+sigma_val*alpha_minus*phi_prime(0)}")
                #print(alpha_minus)
                #print("firstWhile")
            alpha_plus = alpha_minus

            while phi(alpha_plus) <= phi(0) + sigma_val*alpha_plus*phi_prime(0): # Armijo rule: alpha_plus
                alpha_plus = 2*alpha_plus
                #print("secondtWhile")
            
            for p_val in (np.linspace(sigma_val,1-0.01,50)): # loop through alpha values
                while not phi_prime(alpha_minus) >= p_val*phi_prime(0): # condition 2 from slides :)
                    alpha_0 = (alpha_plus + alpha_minus)/2
                    #print("thirdWhile")
                    if phi(alpha_0) <= phi(0) + sigma_val*alpha_0*phi_prime(0): # Armijo rule: alpha_0
                        alpha_minus = alpha_0
                    else:
                        alpha_plus = alpha_0
        # alpha_minus is the inexact line search variable
        
        time.sleep(1)
        print(f"{alpha_minus}")
        return x + alpha_minus*s
