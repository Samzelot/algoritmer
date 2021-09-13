import matplotlib.pyplot as plt
import numpy as np
import util

'''Description: 
    Main Spline class
    '''
class Spline:
    def __init__(self, controlpoints, order = 3, resolution = 100):
        '''Description: 
        Initialisation of the spline class. Inputs are the controlpoints = d_i, order (in our case cubic), resolution = 1/stepsize.
        '''
        self.p = order
        self.controlpoints = np.asarray(controlpoints)
        self.u_knots = np.arange(len(self.controlpoints))
        self.space = np.linspace(0, self.u_knots[-1], resolution)

    def __call__(self):
        return np.array([self.eval(u) for u in self.space])

    
    def eval(self, u: float):
        '''Description: 
        Evaluates the spline at the point u.
        '''
        #i is the index of the 'hot interval'
        i = self.u_knots.searchsorted(u) 

        return self._blossom(u, i, self.p)


    def _blossom(self, u: float, i: int, r: int):
        '''Description: 
        Runs the recursive blossom function. Input u=parameter, i=index of the interval, r=depth of the recursion. 
        Return = p = spline evaluation of parameter u for order r in the recursion.
        '''

        if r == 0:
            return util.clamp_arr(i, self.controlpoints) 
        
        denominator = (util.clamp_arr(i - 1, self.u_knots) - util.clamp_arr(i + self.p - r, self.u_knots))
        alpha = (util.clamp_arr(i - 1, self.u_knots) - u)/denominator if denominator != 0 else 0
        pl = self._blossom(u, i - 1, r - 1)
        pr = self._blossom(u, i, r - 1)
        p = alpha*pr + (1 - alpha)*pl
        return p

    def plot(self):
        '''Description: 
        Plots the spline.
        '''
        plt.plot(*self().T, "-")
        plt.plot(*self.controlpoints.T, "o:")
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
