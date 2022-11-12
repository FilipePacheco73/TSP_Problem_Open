# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:28:14 2022

@author: Z52XXR7

Programa para resolução do Problema do Caixeiro Viajante - Travelsales man Problem -
A fim de estabelecer um Benchmark para verificação de capacidade de processamento - 

Algoritmo de busca local de posição fixa e com aleatoriedade para escapar dos mínimos locais

"""

import numpy as np
from aco_routing.utils.graph import Graph
from aco_routing.dijkstra import Dijkstra
from aco_routing.utils.simulator import Simulator
from aco_routing.aco import ACO

#Main code

N = 10 # tamanho do problema

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)
# np.random.seed(73)

for i in range(N):
    M[i,i] = 1000
    
graph = Graph()
for i in range(N):
    for j in range(N):
        graph.add_edge(i,j, travel_time=M[i,j])

source = 0
destination = 9

aco = ACO(graph)
dijkstra = Dijkstra(graph)

dijkstra_path, dijkstra_cost = dijkstra.find_shortest_path(source, destination)
aco_path, aco_cost = aco.find_shortest_path(source, destination)

print(f"ACO - path: {aco_path}, cost: {aco_cost}")
print(f"Dijkstra - path: {dijkstra_path}, cost: {dijkstra_cost}")
