
from problem import Problem
from solver import NewtonsMethod
import numpy as np

def main():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    #f = lambda x: np.sum(x**2)
    problem = Problem(f)
    solver = NewtonsMethod('cauchy', 1e-3, 1e-5)
    print(solver.solve(problem, np.array([1, 2])))

if __name__ == "__main__":
    main()