#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 21:21:57 2021

@author: pontusgreen
"""


def evaluate_basis(j,u_knots,k):
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
        if k == 0:
            return lambda u: (1 if u_knots[j-1] <= u < u_knots[j] else 0)
        else:
            
            def evalu(u):
                term1 = (u - u_knots[j-1])/(u_knots[j+k-1]-u_knots[j-1])
                term2 = ((u_knots[j+k]- u)/(u_knots[j+k]-u_knots[j]))
                
                a = term1 * evaluate_basis(j,u_knots, k-1)(u)
                b = term2 * evaluate_basis(j+1,u_knots,k-1)(u)
                return  a + b 
    
            return evalu   
