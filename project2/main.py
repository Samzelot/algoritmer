
from project2.stoppingstrategies import CauchyStopping, ResidualStopping
from problem import Problem
from solver import GlobalParams, QuasiNewtonMethod
from stepstrategies import *
from hessianstrategies import *
import numpy as np
import matplotlib.pyplot as plt
import itertools as iter


class LoggingHessian(HessianStrategy):

    def __init__(self):
        self.H_1_strategy = FiniteDifferenceHessian()
        self.H_2_strategy = DFP_rank_2_Hessian()
        self.log = []

    def hessian(self, problem, globals, x):
        H_1 = self.H_1_strategy.hessian(problem, globals, x)
        H_2 = self.H_2_strategy.hessian(problem, globals, x)

        self.log.append((H_1, H_2))
        return H_1

    def get_log(self):
        return self.log

def main():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    #f = lambda x: np.sum((x)**2, )
    problem = Problem(f) #TODO make gradient work

    hessian = LoggingHessian()
    params = GlobalParams(1e-5)
    stop = ResidualStopping(1e-5)
    solver = QuasiNewtonMethod(hessian, DefaultStep(), stop, params)
    val, points = solver.solve(problem, np.array([0, -0.75]), debug=True)
    plot_countour(f, 100, -0.5, 2, -1.5, 4)
    plt.plot(*np.asarray(points).T, ".")
    plt.show()


def plot_countour(f, resolution, x_min, x_max, y_min, y_max):
    Z = np.zeros((resolution, resolution))
    X = np.linspace(x_min, x_max, resolution)
    Y = np.linspace(y_min, y_max, resolution)
    for (i, x), (j, y) in iter.product(enumerate(X), enumerate(Y)):
        Z[i, j] = f(np.array([x, y]))
    
    levels=np.array([1, 3.831, 14.678, 56.234, 215.443, 825.404])
    plt.contour(X, Y, Z.T, levels, cmap='gray')


if __name__ == "__main__":
    main()