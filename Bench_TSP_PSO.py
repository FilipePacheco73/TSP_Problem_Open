# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:25:04 2021

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing PSO - Particle Swarm Optimization

"""

# Preamble - Imports

import numpy as np
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO
from numba import jit
import time

# Main code

N = 100 # Size of the problem
d = 10 # Number of particles - Optimization parameters

# Creating the distances of the problem

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000

@jit() # Function with Numba package - convert into C to run faster
def TSP(x):
    cc = np.copy(x)
    for k in range(d):
        for j in range((N-1)**2):
            for i in range(N-1):
                if cc[k,i] < cc[k,i+1]:
                    temp = cc[k,i]
                    cc[k,i] = cc[k,i+1]
                    cc[k,i+1] = temp

    Top = np.asarray(np.zeros((d,N)), dtype = np.int32)   
    for k in range(d):       
        for i in range(N):
            for j in range(N):
                if cc[k,i] == x[k,j]:
                    Top[k,i] = j
    
    OBJ = np.asarray(np.zeros(d), dtype = np.float32) 
    
    # Objective function
    for k in range(d):
        for i in range(N-1):
            OBJ[k] += M[Top[k,i],Top[k,i+1]]
        
    # print(np.min(OBJ))
    return OBJ

#Set-up hyperparameters 
options = {'c1':0.5,'c2':0.9,'w':0.9}
# options = {'c1':0.5,'c2':0.3,'w':0.9,'k':3,'p':2}

#Create bounds
max_bound = 1000*np.ones(N)
min_bound = -max_bound
bounds = (min_bound, max_bound)

#Call instance of PSO
start = time.time()
optimizer = GlobalBestPSO(n_particles=d, dimensions=N, options=options, bounds=bounds)
cost, pos = optimizer.optimize(TSP,100)

# Results
print(cost)
stop = time.time()
print("Elapsed time", stop-start)
