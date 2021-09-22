import spline as sp
import numpy as np

class InterpolationSpline(sp.Spline):
    def __init__(self, interpolation_points, order = 3, resolution = 100):

        x, y = np.asarray(interpolation_points).T
        u_knots = np.arange(len(interpolation_points))
        de_boor_points = 1
        controlpoints = 2
        sp.Spline.__init__(self, controlpoints, order, resolution)