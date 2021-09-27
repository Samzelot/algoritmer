from abc import ABC, abstractmethod
from hessianstrategies import *
from solver import *
import numpy as np
from problem import Problem


def get_line_problem(problem, finite_difference_step, x, s):
    f_line_x = lambda alpha: x + alpha*s
    return Problem(
        lambda alpha: problem.f(f_line_x(alpha)),
        #lambda alpha: np.atleast_1d(
        #    [problem.g(finite_difference_step)(f_line_x(alpha))@s/np.linalg.norm(s)]
        #)
    )
    
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
        line_problem = get_line_problem(problem, self.line_solver.global_params.finite_difference_step, x, s)
        alpha = self.line_solver.solve(line_problem, np.array([1]))
        return f_line_x(alpha)

class InexactLineStep(StepStrategy):
    def __init__(self, sigma=0.1, rho=0.9):
        self.sigma = sigma
        self.rho = rho

    def step(self, problem, globals, x, s):
        line_problem = get_line_problem(problem, globals.finite_difference_step, x, s)
        phi = line_problem.f
        phi_prime = line_problem.g(globals.finite_difference_step)
        alpha_minus = 2 # random alpha minus start
        
        while not self.armijo_condition(phi, phi_prime, alpha_minus): # Armijo rule: alpha_minus
            alpha_minus = alpha_minus/2

        alpha_plus = alpha_minus

        while self.armijo_condition(phi, phi_prime, alpha_plus): # Armijo rule: alpha_plus
            alpha_plus = 2*alpha_plus
        
        while not self.other_condition(phi, phi_prime, alpha_minus) and abs(alpha_minus - alpha_plus) > 1e-4: # condition 2 from slides :)
            alpha_0 = (alpha_plus + alpha_minus)/2
            if self.armijo_condition(phi, phi_prime, alpha_0): # Armijo rule: alpha_0
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0

        return x + alpha_minus*s
    
    def armijo_condition(self, phi, phi_prime, alpha):
        return phi(alpha) <= phi(0) + self.sigma*alpha*phi_prime(0)

    def other_condition(self, phi, phi_prime, alpha):
        return phi_prime(alpha) >= self.rho*phi_prime(0)
