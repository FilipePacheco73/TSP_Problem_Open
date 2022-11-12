# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:01:25 2022

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing Sci-Py Optimization approach

"""

# Preamble - Imports

import numpy as np
from scipy.optimize import minimize
import time

# Main code

N = 20 # Size of the problem

# Creating the distances of the problem

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000
    
#Optimization parameters 

def objective(c):
    
    cc = np.copy(c)#c.copy()
    for j in range((N-1)**2):
        for i in range(N-1):
            if cc[i] < cc[i+1]:
                temp = cc[i]
                cc[i] = cc[i+1]
                cc[i+1] = temp

    Top = []          
    for i in range(N):
        for j in range(N):
            if cc[i] == c[j]:
                Top.append(j)
    
    # Objective Function
    
    OBJ = float(0)
    for k in range(N-1):
        OBJ += float(M[Top[k],Top[k+1]])            
        
    return OBJ


x0 = np.arange(N)
# x0 = np.random.rand(N)
start = time.time()

# Call the solver into Sci-Py package
res = minimize(objective, x0, method='Nelder-Mead',
               options={'xatol':1e-20,'disp':True,'maxiter':100})

# res = minimize(objective, x0, method='BFGS',
#                options={'disp':True})

done = time.time()

c = res.x
print("Elapsed time", done-start)                 