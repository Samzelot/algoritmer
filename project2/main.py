
from problem import Problem
from solver import Solver

def main():
    f = lambda x: x**2
    problem = Problem(f)
    solver = Solver()
    solver.solve(problem)

if __name__ == "__main__":
    main()