import numpy as np
from scipy.optimize import approx_fprime
from abc import abstractmethod, ABC
from problem import Problem

class GlobalParams:
    def __init__(self, finite_difference_step):
        self.finite_difference_step = finite_difference_step

class Solver(ABC): 
    
    @abstractmethod
    def solve(self, f):
        pass

class QuasiNewtonMethod(Solver):
    
    def __init__(self, hessian_strategy, step_strategy, stop_strategy, global_params):
        self.hessian_strategy = hessian_strategy
        self.step_strategy = step_strategy
        self.stop_strategy = stop_strategy
        self.global_params = global_params
  
    def solve(self, problem, guess, iters=1000, debug=False):
        f = problem.f
        g = problem.g(self.global_params.finite_difference_step)

        if debug:
            points = [guess]

        x = guess
        for i in range(iters):
            H_inv = self.hessian_strategy.hessian(problem, self.global_params, x)
            s = -H_inv@g(x)
            new_x = self.step_strategy.step(problem, self.global_params, x, s)
            
            if debug:
                print(f'iter {i}, stepped to {new_x}')
                points.append(new_x)

            if self.stop_strategy.should_stop(problem, self.global_params, x, new_x):
                if debug:
                    return new_x, points
                else:
                    return new_x

            x = new_x

        raise RuntimeError("Maximum number of itrerations exceeded")
