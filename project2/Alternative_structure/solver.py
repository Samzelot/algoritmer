from stopping_criterion import cauchy_criterion_fulfilled, residual_criterion_fulfilled
from step_strategies import exact_line_search_step, goldstein_line_search_step, no_line_search_step, wolfe_line_search_step
from hessian_inverse import finite_differences, good_broyden, bad_broyden, symmetric_broyden, DFP_rank2, BFGD_rank2
import numpy as np

class Solver:
    def __init__(self, line_search_strategy = "wolfe", hessian_inverse_strategy = "good_broyden", stopping_criterion = "residual"):
        self.line_search_strategy = line_search_strategy
        self.hessian_inverse_strategy = hessian_inverse_strategy
        self.stopping_criterion = stopping_criterion
        self._validate_input_parameters()

    def _validate_input_parameters(self):
        line_search_strategies = ["no_line_search", "exact_line_search", "golstein", "wolfe"]
        hessian_inverse_strategies = ["finite_differences", "good_broyden", "bad_broyden", "symmetric_broyden", "DFP_rank2", "BFGD_rank2"]
        stopping_criterions = ["residual", "cauchy"]

        if self.line_search_strategy not in line_search_strategies:
            raise ValueError("Line search strategy has to be one of:" + str(line_search_strategies))
            
        if self.hessian_inverse_strategy not in hessian_inverse_strategies:
            raise ValueError("Hessian strategy has to be one of:" + str(hessian_inverse_strategies))
        
        if self.stopping_criterion not in stopping_criterions:
            raise ValueError("Stopping criterion has to be one of:" + str(stopping_criterions))
        return

    def solve(self, problem, x0, iters = 1000, h = 1e-5):
        function = problem.function
        gradient = problem.gradient
        x = x0
        for i in range(iters):
            H_inv = self._hessian_inverse()
            s = H_inv*gradient(x)
            x_next = self._step()
            if self._stopping_criterion_fulfilled():
                return x_next
            else:
                x = x_next


    def _hessian_inverse(self):
        strat = self.hessian_inverse_strategy
        if strat == "finite_differences":
            return finite_differences()
        elif strat == "good_broyden":
            return good_broyden()
        elif strat == "bad_broyden":
            return bad_broyden()
        elif strat == "symmetric_broyden":
            return symmetric_broyden()
        elif strat == "DFP_rank2":
            return DFP_rank2()
        elif strat == "BFGD_rank2":
            return BFGD_rank2()
        else:
            raise Exception

    def _step_strategies(self): #alpha
        strat = self.line_search_strategy
        if strat == "no_line_search":
            return no_line_search_step()
        elif strat == "exact_line_search":
            return exact_line_search_step()
        elif strat == "goldstein":
            return goldstein_line_search_step()
        elif strat == "wolfe":
            return wolfe_line_search_step()
        else:
            raise Exception

    def _stopping_criterion_fulfilled(self):
        criterion = self.stopping_criterion
        if criterion == "residual":
            return residual_criterion_fulfilled()
        elif criterion == "cauchy":
            return cauchy_criterion_fulfilled()
        else:
            raise Exception
    



        

       
