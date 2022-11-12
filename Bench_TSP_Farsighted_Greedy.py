# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:46:09 2022

@author: Filipe Pacheco

Code to solve Travel Salesman Problem
Main objective to establish a benchmarking for processing capacity verification 

Utilizing Farsighted Greedy Algorithm with different starting point

"""

# Preamble - Imports
import numpy as np
from numba import jit
import time
from itertools import permutations, combinations

# Main code

N = 10 # Size of the problem

# Creating the distances of the problem
np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000
    
@jit() # Function with Numba package - convert into C to run faster
def func_objective(Top_aux,n):
    OBJ_aux = 0
    for k in range(n):
        OBJ_aux += M[Top_aux[k],Top_aux[k+1]]
    return OBJ_aux

# Creating the permutation set list
def permutations_(set_permutation,cut_front):
    l = list(permutations(set_permutation,cut_front))
    l = np.array(l)
    return l

# Testable solution with Farsighted Greedy Algorithm
def objective(N):
    ASW = np.zeros(N+1)
    TP = []
    set_permutation = np.arange(N).tolist()
    
    for ii in range(N):
        cut_front = min(6,N-ii) # Choosable parameter
        for i in TP:
            if i in set_permutation:
                set_permutation.remove(i)        
        # sub_set_permutation = set_permutation.copy()
        # random.shuffle(sub_set_permutation)
        # sub_set_permutation = sub_set_permutation[:min(len(set_permutation),19)]
                                                        
        l = permutations_(set_permutation,cut_front)
        # l = permutations_(sub_set_permutation,cut_front)
        print(len(l),ii+1)

        MG_aux = 1000000
        for i in range(len(l)):
            Top_aux = np.array(l[i])
            Top_aux = np.insert(Top_aux,0,np.array(TP))
                        
            OBJ_aux = func_objective(Top_aux,cut_front+ii-1)
            
            if OBJ_aux < MG_aux:
                MG_aux = OBJ_aux
                # print(OBJ_aux,ii,Top_aux)   
                TP_aux = Top_aux
                
        TP.append(TP_aux[ii])
        # print(MG_aux,ii+1)

    MG = 0
    for k in range(N-1):
        MG += M[TP[k],TP[k+1]]
                
    ASW[:N] = TP
    ASW[N] = MG
    return ASW
    
# Results
start = time.time()
ASW = objective(N)
done = time.time()
MG = ASW[N]
print("\nObjective Function:",MG)
print("Elapsed Time:", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)