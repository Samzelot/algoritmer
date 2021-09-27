import numpy as np
from problem import Problem
from solver import Solver

def main():
    f = lambda x: x[0]**2 + x[1]**3 + x[2]**4
    problem = Problem(f, h=1e-10)
    solver = Solver(hessian_inverse_strategy="bad_broyden")
    print(problem.gradient([1,1,1]))



if __name__ == "__main__":
    main()