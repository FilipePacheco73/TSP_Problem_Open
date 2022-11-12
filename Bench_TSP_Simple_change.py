# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:48:01 2022

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing Simplex Algorithm to find the semi-optimal solution

"""

# Preamble - Imports

import numpy as np
from numba import jit
import time

# Main code

N = 100 # Size of the problem

# Creating the distances of the problem

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000
    
#Optimization parameters 

Top = np.arange(N)
# np.random.shuffle(Top)
# print(Top)
# print(len(np.unique(Top)))
Top = np.asarray(Top, dtype=np.int32)
TP = np.zeros(N, dtype=np.int32)
OG = np.ones(N, dtype=np.float64)*100000000000
RANK = np.zeros(N, dtype=np.int32)
ASW = np.zeros(N+1)

# Optimization Simplex algorithm
@jit() # Function with Numba package - convert into C to run faster
def objective(Top,TP,OG,RANK,ASW):
    
    MG = 100000000000
    TOPP = Top.copy()    
    
    for kk in range(10000):
        Top = TOPP.copy()
        for i in range(np.random.randint(2)):
            ii = np.random.randint(N)
            ij = np.random.randint(N)
            # ik = np.random.randint(N)
            # il = np.random.randint(N)
        
            temp = Top[ij]
            Top[ij] = Top[ii]
            Top[ii] = temp
            
            # temp = Top[ij]
            # Top[ij] = Top[ii]
            # Top[ii] = Top[ik] 
            # Top[ik] = temp
            
            # temp = Top[ij]
            # Top[ij] = Top[ii]
            # Top[ii] = Top[ik] 
            # Top[ik] = Top[il]
            # Top[il] = temp
        
        # Objective Function
        
        OBJ = float(0)
        for k in range(N-1):
            OBJ += float(M[Top[k],Top[k+1]])       
                    
        if OBJ < MG:
            TOPP = Top.copy()
            TP = Top.copy()
            MG = OBJ
            print(MG,kk)

    ASW[:N] = TP
    ASW[N] = MG
    return ASW

start = time.time()
ASW = objective(Top,TP,OG,RANK,ASW)
done = time.time()
MG = ASW[N]
print(MG)
print("Elapsed time", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)