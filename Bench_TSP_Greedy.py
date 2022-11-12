# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 07:42:43 2021

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing Greedy Algorithm with different starting point

"""

# Preamble - Imports

import numpy as np
import time

# Main code

N = 100 # Size of the problem

# Creating the distances of the problem
np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000
  
# Optimization Greedy Algorithm
def objective(N):
    Top = np.arange(N)
    TP = np.asarray(np.zeros(N), dtype=np.int32)
    MG = 100000000
    ASW = np.zeros(N+1)
    
    
    for kk in range(N):
        visited = -1*np.ones(N,np.int32)
        visited[0] = kk
                
        for i in range(N-1):
            dist_aux = 10000000
            for j in range(N):
                if M[visited[i],j] < dist_aux:
                    if j not in visited:
                        # print(j)
                        dist_aux = M[visited[i],j]
                        next_ = j

            
            visited[i+1] = next_
        
        # Top = np.array(visited)
        Top = visited
        # print(visited)

        # Objective function
        OBJ = float(0)
        for k in range(N-1):
            OBJ += float(M[Top[k],Top[k+1]])  
            
        if OBJ < MG:
            print(OBJ,kk)
            MG = OBJ
            TP = Top.copy()
        
    ASW[:N] = TP
    ASW[N] = MG
    return ASW

# Results

start = time.time()
ASW = objective(N)
done = time.time()
MG = ASW[N]
print(MG)
print("Elapsed time", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)