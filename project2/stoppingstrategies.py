from abc import ABC, abstractmethod
import numpy as np

class StoppingStrategy(ABC):
    @abstractmethod
    def should_stop(self, problem, globals, x, new_x):
        return

class CauchyStopping(StoppingStrategy):
    def __init__(self, error):
        self.error = error

    def should_stop(self, _1, _2, x, new_x):
        return np.linalg.norm(x-new_x) < self.error

class ResidualStopping(StoppingStrategy):
    def __init__(self, error):
        self.error = error

    def should_stop(self, problem, globals, x, new_x):
        g = problem.g(globals.finite_difference_step)
        return np.linalg.norm(g(new_x))< self.error

class CombinedStopping(StoppingStrategy):
    def __init__(self, stopping_strategies, combiner):
        self.stopping_strategies = stopping_strategies
        self.combiner = combiner
    
    def should_stop(self, problem, x, new_x):
        return self.combiner(s.should_stop(problem, globals, x, new_x) for s in self.stopping_strategies)