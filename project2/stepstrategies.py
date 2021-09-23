from abc import ABC, abstractmethod
from hessianstrategies import FiniteDifferenceHessian
from solver import QuasiNewtonMethod
import numpy as np
from problem import Problem
from scipy.optimize import approx_fprime
import time

class StepStrategy(ABC):
    @abstractmethod
    def step(self, problem, globals, x, s):
        return

class DefaultStep(StepStrategy):
    def step(self, _1, _2, x, s):
        return x + s

class ExactLineStep(StepStrategy):
    def __init__(self, line_solver):
        self.line_solver = line_solver

    def step(self, problem, _, x, s):
        f_line_x = lambda alpha: x + alpha*s
        line_problem = Problem(
            lambda alpha: problem.f(f_line_x(alpha)),
            lambda alpha: np.atleast_1d(
                [problem.g(self.line_solver.global_params.finite_difference_step)(f_line_x(alpha))@s/np.linalg.norm(s)]
            ),
        )
        alpha = self.line_solver.solve(line_problem, np.array([1]))
        return f_line_x(alpha)

class InexactLineStep(StepStrategy):
    def __init__(self, sigma=0.1, rho=0.9):
        self.sigma = sigma
        self.rho = rho

    def step(self, problem, globals, x, s):
        phi = lambda alpha: problem.f(x + alpha*s) # phi
        #phi_prime = lambda alpha: problem.g(self.finite_differences_step)(x + alpha*s)@s/np.linalg.norm(s) # phi prime calculation
        phi_prime = lambda alpha : approx_fprime(x+ alpha*s, problem.f, globals.finite_difference_step)@(s/np.linalg.norm(s)) 
        alpha_minus = 2 # random alpha minus start
        
        while not self.armijo_condition(phi, phi_prime, alpha_minus): # Armijo rule: alpha_minus
            alpha_minus = alpha_minus/2
            #print(f"phi_prime: {phi_prime(0)}")
            #print(f" {phi(alpha_minus)} <= {phi(0)+sigma_val*alpha_minus*phi_prime(0)}")
            #print(alpha_minus)
            #print("firstWhile")

        alpha_plus = alpha_minus

        while self.armijo_condition(phi, phi_prime, alpha_plus): # Armijo rule: alpha_plus
            alpha_plus = 2*alpha_plus
            #print("secondtWhile")
        
        while not self.other_condition(phi, phi_prime, alpha_minus) and abs(alpha_minus - alpha_plus) > 1e-4: # condition 2 from slides :)
            alpha_0 = (alpha_plus + alpha_minus)/2
            #print(f'- = {alpha_minus}, + = {alpha_plus}, 0 = {alpha_0}, phi_prime(alpha_minus) = {phi_prime(alpha_minus)}, phi_prime(0) = {phi_prime(0)*self.rho}')
            #print("thirdWhile")
            if self.armijo_condition(phi, phi_prime, alpha_0): # Armijo rule: alpha_0
                alpha_minus = alpha_0
                #print("set plus")
            else:
                #print("set minus")
                alpha_plus = alpha_0
        # alpha_minus is the inexact line search variable
        
        #print(f"{alpha_minus}")
        return x + alpha_minus*s
    
    def armijo_condition(self, phi, phi_prime, alpha):
        return phi(alpha) <= phi(0) + self.sigma*alpha*phi_prime(0)

    def other_condition(self, phi, phi_prime, alpha):
        return phi_prime(alpha) >= self.rho*phi_prime(0)
