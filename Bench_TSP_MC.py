# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:48:01 2022

@author: Z52XXR7

Programa para resolução do Problema do Caixeiro Viajante - Travelsales man Problem -
A fim de estabelecer um Benchmark para verificação de capacidade de processamento - 

Algoritmo de busca local de posição fixa e com aleatoriedade para escapar dos mínimos locais

Abordagem voltada a aplicação de visitas de Monte Carlo

"""
#Preamble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit, vectorize, prange
import time

#Main code

N = 20 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
# np.random.seed(73)

for i in range(N):
    M[i,i] = 1000
gamma = .9

@jit()   
def repetition(N):
    MG = 100000000
    # RANK = 100000000*np.ones((N,N), dtype=np.float32)
    RANK = 100000000*np.random.rand(N,N)
    Top = np.arange(N, dtype=np.int32)
    TP = Top.copy()
    for ii in range(100000*N):

        np.random.shuffle(Top)
        
        OBJ = float(0)
        for k in range(N-1):
            OBJ += float(M[Top[k],Top[k+1]])  
            
        for i in range(N):
            RANK[Top[i],i] = OBJ + gamma*RANK[Top[i],i] # TD - Learning
            # if OBJ < RANK[Top[i],i]:
            #     RANK[Top[i],i] = OBJ
                
    

    return RANK

start = time.time()
RANK = repetition(N)
done = time.time()
print("elapsed time", done-start)
print(np.min(RANK))
RANK = 100000000*np.random.rand(N,N)

@jit()
def objective(RANK,M,N):
    Top = np.arange(N)
    NN = 3*N
    # np.random.shuffle(Top)
    Top = np.asarray(Top, dtype=np.int32)
    TOP = np.reshape(np.zeros(NN*N, dtype=np.int32),(NN,N))
    TP = np.zeros(N, dtype=np.int32)
    OG = np.ones(NN, dtype=np.float64)*100000000000
    ASW = np.zeros(N+1)
    MG = 100000000000
    for kk in range(10000):
        TOPP = Top.copy()
        OG = np.ones(NN)*100000000000
        for jj in range(NN):
            ii = np.random.randint(N)    
            ij = np.random.randint(N)
            
            OBJ = float(0)
            for k in range(N-1):
                OBJ += float(RANK[Top[k],Top[k+1]])               
                
            OG[jj] = float(OBJ)
            TOP[jj,:] = Top.copy()
            
            if OG[jj] < MG:
                MG = OG[jj]
                OBJ = float(0)
                for k in range(N-1):
                    OBJ += float(M[Top[k],Top[k+1]])
                # print(OBJ)
                print(MG, OBJ, kk)
                TP = Top.copy()
                TOPP = Top.copy()
                
            Top = TOPP.copy()
            temp = Top[ij]
            Top[ij] = Top[ii]
            Top[ii] = temp

        
        if min(OG) == OG[0]:
            aux = np.random.randint(max([1,round(.5*NN,0)]))
            Top = TOP[aux+np.argmin(OG[aux:])]
            # aux = np.argsort(OG)
            # Top = TOP[aux[np.random.randint(0.6*NN)]]
            # Top = TOP[np.argsort(OG)[1]]
        else:
            Top = TOP[np.argmin(OG)]

    ASW[:N] = TP
    ASW[N] = MG
    return ASW

start = time.time()
ASW = objective(RANK,M,N)
done = time.time()
MG = ASW[N]
print(MG)
print("elapsed time", done-start)
ASW = np.delete(ASW,N)
TP = np.asarray(ASW, dtype=np.int32)
print(len(np.unique(TP)))
















# # @jit()
# def objective(N):
#     Top = np.arange(N)
#     TP = np.asarray(np.zeros(N), dtype=np.int32)
#     MG = 100000000
#     ASW = np.zeros(N+1)
    
    
#     for kk in range(N):
#         visited = -1*np.ones(N,np.int32)
#         visited[0] = kk
                
#         for i in range(N-1):
#             dist_aux = 10000000
#             for j in range(N):
#                 if RANK[visited[i],j] < dist_aux:
#                     if j not in visited:
#                         # print(j,i)
#                         if np.random.randint(100) < 50:
#                             dist_aux = RANK[visited[i],j]
#                         next_ = j
#             visited[i+1] = next_
        
#         # Top = np.array(visited)
#         Top = visited
#         # print(visited)

#         OBJ = float(0)
#         for k in range(N-1):
#             OBJ += float(M[Top[k],Top[k+1]])  
            
#         if OBJ < MG:
#             print(OBJ,kk)
#             MG = OBJ
#             TP = Top.copy()
        
#     ASW[:N] = TP
#     ASW[N] = MG
#     return ASW

# start = time.time()
# ASW = objective(N)
# done = time.time()
# MG = ASW[N]
# print(MG)
# print("elapsed time", done-start)
# ASW = np.delete(ASW,N)
# TP = np.asarray(ASW, dtype=np.int32)
# print(len(np.unique(TP)))