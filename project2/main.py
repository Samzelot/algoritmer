
from stoppingstrategies import *
from problem import Problem
from solver import *
from stepstrategies import *
from hessianstrategies import *
import numpy as np
import matplotlib.pyplot as plt
import itertools as iter
from chebyquad_problem import chebyquad, gradchebyquad

class LoggingHessian(HessianStrategy):

    def __init__(self):
        self.H_1_strategy = FiniteDifferenceHessian()
        self.H_2_strategy = BFGS_rank_2_Hessian()
        self.log = []

    def hessian(self, problem, globals, x):
        H_1 = self.H_1_strategy.hessian(problem, globals, x)
        H_2 = self.H_2_strategy.hessian(problem, globals, x)

        self.log.append((H_1, H_2))
        return H_1

    def get_log(self):
        return self.log

def main():
    task_5()
    task_7()
    task_10()
    task_12()


def task_12():
    #Problem
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    #Solution
    problem = Problem(f) 
    hessian = LoggingHessian()
    params = GlobalParams(1e-5)
    stop = CauchyStopping(1e-3)
    line_solver = QuasiNewtonSolver(FiniteDifferenceHessian(), DefaultStep(), stop, params)
    solver = QuasiNewtonSolver(hessian, ExactLineStep(line_solver), stop, params)
    val, points = solver.solve(problem, np.array([0, -0.75]), debug=True)
    diff_hessians=[]
    for i in range(len(hessian.get_log())):
        diff_hessians.append(sum(sum(abs(hessian.get_log()[i][0]-hessian.get_log()[i][1]))))
    plt.figure(1)
    plt.plot(diff_hessians)

    plt.figure(2)
    plot_countour(f, 100, -0.5, 2, -1.5, 4)
    plt.plot(*np.asarray(points).T, ".")
    plt.show()
    
def task_10():

    #Problem
    problem = Problem(chebyquad, gradchebyquad)

    #Solution
    hessian = FiniteDifferenceHessian()
    params = GlobalParams(finite_difference_step=1e-5)
    stop = CauchyStopping(error=1e-3)
    line_solver = QuasiNewtonSolver(hessian, DefaultStep(), stop, params)
    solver = QuasiNewtonSolver(hessian, ExactLineStep(line_solver), stop, params)
    guess  = np.linspace(0,1,11)
    val, points = solver.solve(problem, guess, debug=True)
    print(f'min_coord: {val}, min_val: {chebyquad(val)}')

def task_5():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    problem = Problem(f)

    hessian = FiniteDifferenceHessian()
    params = GlobalParams(1e-5)
    stop = CauchyStopping(1e-5)
    line_solver = QuasiNewtonSolver(hessian, DefaultStep(), stop, params)
    solver = QuasiNewtonSolver(hessian, ExactLineStep(line_solver), stop, params)
    val, points = solver.solve(problem, np.array([0, -0.75]), debug=True)
    plot_countour(f, 100, -0.5, 2, -1.5, 4)
    plt.plot(*np.asarray(points).T, ".")
    plt.show()

def task_7():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    problem = Problem(f)

    hessian = FiniteDifferenceHessian()
    params = GlobalParams(1e-5)
    stop = CauchyStopping(1e-5)
    solver = QuasiNewtonSolver(hessian, InexactLineStep(0.1, 0.01), stop, params)
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