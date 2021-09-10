import numpy as np
import matplotlib.pyplot as plt
import util

def evaluate_basis(j,u_knots,p):
        """
        Description
        -----------
        Takes as input a knot sequence u, and an index j that returns a function that
        evaluates the j:th B-spline basis function N^3_j.

        Parameters
        -----------
        j : int
            index point
        u_knots : array
            knot sequence
        k : int
            current degree

        Returns
        ------
        evalf: lambda function?
            Function that evaluates the j:th B-spline basis at index j
        """
        
        """
        for val in u:
            deg = 3
            current_knot_index = find_knot_index(val, u)
            
            #start recursion if we u value is in correct knot index
            if current_knot_index == j:
        """
        #recursion base
        if p == 0:
            return lambda u: (1 if util.clamp_arr(j-1,u_knots) <= u < util.clamp_arr(j,u_knots) else 0)
        else:
            
            def evalu(u):
                den1 = util.clamp_arr(j+p-1,u_knots)-util.clamp_arr(j-1,u_knots)
                term1 = (u - util.clamp_arr(j-1,u_knots))/den1 if den1 != 0 else 0
                den2 = util.clamp_arr(j+p,u_knots)-util.clamp_arr(j,u_knots)
                term2 = (util.clamp_arr(j+p,u_knots)- u)/den2 if den2 != 0 else 0
                
                a = term1 * evaluate_basis(j,u_knots, p-1)(u)
                b = term2 * evaluate_basis(j+1,u_knots,p-1)(u)
                if j==len(u_knots)-1 and u==u_knots[-1]: 
                    return 1
                return  a + b 
            return evalu
            
def create_interval(j):
    """
    Description
    ------------
    Find interval that we are going to sum through and returns it
    -----------
    j : int
        index point

    Returns
    ------
    interval from j-2 to j+1
    """
    return np.arange(j-3, j+1)
        
def evaluate_spline_basis(j,u,u_knots,controlpoints,p):
    """
    Description
    ------------
    Evaluates S(u) by recursive defintion and summing up intervals j = I-2 to I+1
    
    Parameters
    -----------
    j : int
        index point
    u : float
        u value to be evaluated
    u_knots : array
        knot sequence
    controlpoints: array
        controlpoints
    p : int
        current degree

    Returns
    ------
    s: array (float, float)
        Evaluation of the spline
    """
    s = np.zeros(2)
    I = create_interval(j)
    
    for i in I:
        #function that evaluates 
        N_i = evaluate_basis(i,u_knots,p)
        s += util.clamp_arr(i, controlpoints)*N_i(u)

    return s


if __name__ == "__main__":
    
    knots=10
    
    x = np.linspace(0, knots,1000)
    u_knots = np.arange(knots+1)

    for i in range(knots+3):
        N = evaluate_basis(i - 2, u_knots, 3)
        y = [N(j) for j in x]
        plt.plot(x, y)
    plt.show()