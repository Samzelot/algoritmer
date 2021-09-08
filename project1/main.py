import numpy as np
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt

class Spline:
    def __init__(self, controlpoints):
        self.controlpoints = controlpoints
        self.u = list(range(len(self.controlpoints)))
        print(self.u)
        # u : array , knot sequence
        pass

    def __call__(self, segment_resolution):
        spline = []
        N = len(self.controlpoints)
        for i in range(N - 1):
            leftmost = max(i - 1, 0)
            rightmost = min(i + 2, N - 1)
            points = self.controlpoints[leftmost:rightmost + 1]
            for j in range(segment_resolution):
                u = self.u[i] + j/segment_resolution*(self.u[i + 1] - self.u[i])
                a = self.alpha(u, self.u[leftmost], self.u[rightmost])
                spline.append(self.blossom(a, points))
        return spline

    def plot(self):
        x, y = zip(*self.__call__(10))
        plt.plot(x, y, "x-")
        x, y = zip(*self.controlpoints)
        plt.plot(x, y, "*")
        plt.show()

    def blossom(self, alpha: float, points: "list[tuple[float, float]]") -> "tuple[float, float]":
        N = len(points)
        if N > 2:
            split = math.floor(N/2)
            points = [self.blossom(alpha, points[:split + 1]), self.blossom(alpha, points[split:])]

        (p1x, p1y), (p2x, p2y) = points[0], points[1]
        x = p1x*alpha + p2x*(1 - alpha)
        y = p1y*alpha + p2y*(1 - alpha)
        return (x, y)
            

    def alpha(self, u: float, left: float, right: float) -> float:
        return (right - u)/(right - left)

    def evaluate_basis(j,u):
        """
        Description
        -----------
        Takes as input a knot sequence u, and an index j that returns a function that
        evaluates the j:th B-spline basis function N^3_j.

        Parameters
        -----------
        j : int
            index point
        u : array
            knot sequence

        Returns
        ------
        evalf: lambda function?
            Function that evaluates the j:th B-spline basis at index j
        """

def main():
    spline = Spline([
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (0.5, 0)
    ])
    spline.plot()

if __name__ == "__main__":
    main()
