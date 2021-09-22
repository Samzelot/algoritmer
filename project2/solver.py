import numpy as np
from scipy.optimize import approx_fprime
from abc import abstractmethod, ABC
from problem import Problem

class Solver(ABC): 
    
    @abstractmethod
    def solve(self, f):
        pass

class QuasiNewtonMethod(Solver):
    
    def __init__(self, hessian_strategy, step_strategy, stopping_criteria, stopping_error, finite_differences_step):
        self.hessian_strategy = hessian_strategy
        self.step_strategy = step_strategy
        self.stopping_error = stopping_error
        self.stopping_criteria = stopping_criteria
        self.finite_differences_step = finite_differences_step
        
    def solve(self, problem, guess, iters=1000, debug=False):
        f = problem.f
        g = problem.g(self.finite_differences_step)
        #TODO: add attribute for epsilon in approx_fprime function
        if debug:
            points = [guess]
        
        x = guess
        for i in range(iters):
            H_inv = self.hessian_strategy.hessian(problem, x)
            s = -H_inv@g(x) # calculate the next step
            new_x = self.step_strategy.step(problem, x, s)
            
            if debug:
                print(f'iter {i}, stepped to {new_x}')
                points.append(new_x)

            stop = False
            if self.stopping_criteria.lower() == "cauchy":
                if np.linalg.norm(x-new_x) < self.stopping_error: #check the cauchy stopping criteria
                    stop = True
            elif self.stopping_criteria.lower() == "residual":
                if np.linalg.norm(g(new_x))< self.stopping_error:
                    stop = True
            else:
                raise ValueError("Not recognized stopping criterion")

            if stop:
                if debug:
                    return new_x, points
                else:
                    return new_x

            x = new_x

        return x, points
