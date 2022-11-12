# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 07:40:18 2021

@author: Z52XXR7

Programa para resolução do Problema do Caixeiro Viajante - Travelsales man Problem -
A fim de estabelecer um Benchmark para verificação de capacidade de processamento

"""
#Preamble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#Optimization Libraries 
from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model
from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior
from geneticalgorithm2 import Population_initializer # for creating better start population
from geneticalgorithm2 import np_lru_cache # for cache function (if u want)
from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want
from geneticalgorithm2 import Callbacks # simple callbacks
from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks
from OppOpPopInit import OppositionOperators
from numba import jit, vectorize

#Main code

N = 20 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
np.random.seed()


for i in range(N):
    M[i,i] = 1000
    

@jit()
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
    
    # Top = np.argsort(c)
    # Top = np.flip(Top)
    
    OBJ = 0
    
    for i in range(N-1):
        OBJ += M[Top[i],Top[i+1]]
     

    return OBJ 


varbound = np.array([[0,100]]*N)
model = ga(objective, dimension = N, 
                 variable_type='real', 
                 variable_boundaries = varbound,
                 function_timeout = 30,
                 algorithm_parameters={'max_num_iteration': 10000,
                                       'population_size': 5,                                      
                                       'mutation_probability':0.1,
                                       'elit_ratio': 0.1,
                                       'crossover_probability': 0.7,
                                       'parents_portion': 0.5,
                                       'crossover_type':'two_point',
                                       'mutation_type': 'uniform_by_center',
                                       'selection_type': 'roulette',
                                       'max_iteration_without_improv':None}
            )

model.run(
    no_plot = False,
    disable_progress_bar = False,
    set_function = None,#ga.set_function_multiprocess(objective, n_jobs = 2), 
    apply_function_to_parents = False, 
    #start_generation = "Benchmark_TSP.npz",
    start_generation = {'variables': None, 'scores': None},
    studEA = True,
    mutation_indexes = None,
    init_creator = None,
    init_oppositors = None,
    duplicates_oppositor = None,
    remove_duplicates_generation_step = None,
    revolution_oppositor = OppositionOperators.Continual.quasi(minimums = varbound[:,0], maximums = varbound[:, 1]),
    revolution_after_stagnation_step = 100000,
    revolution_part = .3,
    population_initializer = Population_initializer(select_best_of = 1, local_optimization_step = 'never', local_optimizer = None),
    stop_when_reached = None,
    callbacks = [],
    middle_callbacks = [],
    time_limit_secs = None, 
    save_last_generation_as = "Benchmark_TSP.npz",
    seed = None
    )

#Results Representation
X = model.output_dict['variable']
XX = X.copy()

for j in range((N-1)**2):
    for i in range(N-1):
        if XX[i] < XX[i+1]:
            temp = XX[i]
            XX[i] = XX[i+1]
            XX[i+1] = temp
            
Top = []                
for i in range(N):
    for j in range(N):
        if XX[i] == X[j]:
            Top.append(j)
            
print(Top)