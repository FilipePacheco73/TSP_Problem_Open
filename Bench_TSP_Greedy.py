# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 07:42:43 2021

@author: Z52XXR7

Programa para resolução do Problema do Caixeiro Viajante - Travelsales man Problem -
A fim de estabelecer um Benchmark para verificação de capacidade de processamento - 

Algoritmo de busca local de posição fixa e com aleatoriedade para escapar dos mínimos locais

"""
#Preamble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit, vectorize, prange
import time

#Main code

N = 100 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
# np.random.seed(73)

for i in range(N):
    M[i,i] = 1000
    
# @jit()
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
                # if np.random.randint(100) < 99:
                    # dist_aux = 1000000
                if M[visited[i],j] < dist_aux:
                    if j not in visited:
                        # print(j)
                        dist_aux = M[visited[i],j]
                        next_ = j

            
            visited[i+1] = next_
        
        # Top = np.array(visited)
        Top = visited
        # print(visited)

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

start = time.time()
ASW = objective(N)
done = time.time()
MG = ASW[N]
# print(MG)
print("elapsed time", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)
print(len(np.unique(TP)))