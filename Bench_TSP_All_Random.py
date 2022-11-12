# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:35 2022

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing random sequence choices

"""

# Preamble - Imports

import numpy as np
from numba import jit
import time

# Main code

N = 1000 # Size of the problem

# Creating the distances of the problem
np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 100000
    
# Optimization Parameters

Top = np.asarray(np.arange(N), dtype=np.int32)
TP = np.zeros(N, dtype=np.int32)
ASW = np.zeros(N+1)

@jit() # Function with Numba package - convert into C to run faster
def objective(Top,TP,ASW,N):
    MG = 100000000000
    for kk in range(1000000):
            
        np.random.shuffle(Top)
        # Objective function
        OBJ = float(0)
        for k in range(N-1):
            OBJ += float(M[Top[k],Top[k+1]])               
            
            
        if OBJ < MG:
            MG = OBJ
            print(MG,kk)
            TP = Top.copy()
                
    ASW[:N] = TP
    ASW[N] = MG
    return ASW

# Results
start = time.time()
ASW = objective(Top,TP,ASW,N)
done = time.time()
MG = ASW[N]
stop = time.time()
print("Elapsed time", done-start)