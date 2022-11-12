# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:46:09 2022

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
from itertools import permutations, combinations


#Main code

N = 20 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
# np.random.seed(73)

for i in range(N):
    M[i,i] = 1000
    
@jit()
def func_objective(Top_aux,n):
    OBJ_aux = 0
    for k in range(n):
        OBJ_aux += M[Top_aux[k],Top_aux[k+1]]
    return OBJ_aux

def permutations_(set_permutation,cut_front):
    l = list(permutations(set_permutation,cut_front))
    l = np.array(l)
    return l

# @jit()
def objective(N):
    ASW = np.zeros(N+1)
    TP = []
    set_permutation = np.arange(N).tolist()
    
    for ii in range(N):
        cut_front = min(6,N-ii)
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
    
start = time.time()
ASW = objective(N)
done = time.time()
MG = ASW[N]
print("\nObjective Function:",MG)
print("Elapsed Time:", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)
# print(len(np.unique(TP)))
