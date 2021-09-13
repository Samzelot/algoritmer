import unittest 
import spline as sp
import basis
import matplotlib.pyplot as plt
import numpy as np

class TestSpline(unittest.TestCase): 
    

    def test_spline_basis(self):
        '''Description: 
        Tests if the de Boor algorithm gives the same spline as the d_i*N method.
        '''
        spline = sp.Spline([
            (-12.73564, 9.03455),
            (-26.77725, 15.89208),
            (-42.12487, 20.57261),
            (-15.34799, 4.57169),
            (-31.72987, 6.85753),
            (-49.14568, 6.85754),
            (-38.09753, -1e-05),
            (-67.92234, -11.10268),
            (-89.47453, -33.30804),
            (-21.44344, -22.31416),
            (-32.16513, -53.33632),
            (-32.16511, -93.06657),
            (-2e-05, -39.83887),
            (10.72167, -70.86103),
            (32.16511, -93.06658),
            (21.55219, -22.31397),
            (51.377, -33.47106),
            (89.47453, -33.47131),
            (15.89191, 0.00025),
            (30.9676, 1.95954),
            (45.22709, 5.87789),
            (14.36797, 3.91883),
            (27.59321, 9.68786),
            (39.67575, 17.30712)   
        ])

        spline_points = spline()
        test_array=[]
        for i, u in enumerate(spline.space):
            s_val = basis.evaluate_spline_basis(spline.u_knots.searchsorted(u), u, spline.u_knots, spline.controlpoints, spline.p) 
            test_array.append(s_val)
            self.assertAlmostEqual(spline_points[i][0], s_val[0])
            self.assertAlmostEqual(spline_points[i][1], s_val[1])

    def test_basis_sum_to_one(self):
        '''Description: 
        Tests if all the spline basis functions sums to 1.
        '''
        u_knots = np.arange(10)
        sum_n = lambda u: sum([basis.evaluate_basis(i-2, u_knots, 3)(u) for i in range(len(u_knots)+2)])
        for j in np.linspace(0,9,50):
            self.assertAlmostEqual(sum_n(j), 1)
        
            

if __name__ == '__main__': 
    unittest.main()