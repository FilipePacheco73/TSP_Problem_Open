# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:35 2022

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

N = 1000 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
# np.random.seed(73)

for i in range(N):
    M[i,i] = 100000
    
# Otimização
Top = np.asarray(np.arange(N), dtype=np.int32)
TP = np.zeros(N, dtype=np.int32)
ASW = np.zeros(N+1)

@jit()
def objective(Top,TP,ASW,N):
    MG = 100000000000
    for kk in range(1000000):
            
        np.random.shuffle(Top)
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


start = time.time()
ASW = objective(Top,TP,ASW,N)
done = time.time()
MG = ASW[N]
stop = time.time()
print("elapsed time", done-start)