import matplotlib.pyplot as plt
import numpy as np

def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

def clamp_arr(i, arr):
     return arr[clamp(i, 0, len(arr) - 1)]

class Spline:
    def __init__(self, controlpoints, order = 3, resolution = 100):
        self.p = order
        self.controlpoints = np.array([[*p] for p in controlpoints])
        self.u = np.arange(len(self.controlpoints))
        self.space = np.linspace(0, max(self.u), resolution)

    def __call__(self):
        return np.array([self.eval(u) for u in self.space])

    def eval(self, u: float):
        i = self.u.searchsorted(u)
        return self.blossom(u, i, self.p)

    def blossom(self, u: float, i: int, r: int):
        if r == 0:
            return clamp_arr(i, self.controlpoints)
        
        denominator = (clamp_arr(i - 1, self.u) - clamp_arr(i + self.p - r, self.u))
        alpha = (clamp_arr(i - 1, self.u) - u)/denominator if denominator != 0 else 0
        pl = self.blossom(u, i - 1, r - 1)
        pr = self.blossom(u, i, r - 1)
        p = alpha*pr + (1 - alpha)*pl
        return p
    
    def plot(self):
        s = self().T
        plt.plot(s[0], s[1], "-")
        c = self.controlpoints.T
        plt.plot(c[0], c[1], "o:")
        plt.show()

def main():
    spline = Spline([
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (1, 2),
        (2, 2),
        (2, 3),
        (1, 3)
    ])
    spline.plot()

if __name__ == "__main__":
    main()
