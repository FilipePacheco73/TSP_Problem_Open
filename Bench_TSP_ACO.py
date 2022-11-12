# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:28:14 2022

@author: Filipe Pacheco

Code to find the minimum distance into two point in a set of city
Main objective to establish a benchmarking for processing capacity verification 

Utilizing Ant Colony Optimisation & Dijkstra

"""

# Preamble - Imports

import numpy as np
from aco_routing.utils.graph import Graph
from aco_routing.dijkstra import Dijkstra
from aco_routing.aco import ACO

# Main code

N = 10 # Size of the problem

# Creating the distances of the problem

np.random.seed(73)
M = np.random.rand(N,N)
M = np.matmul(M,M.T)

for i in range(N): # Avoid revisit the same city
    M[i,i] = 1000
    
# Creating graph

graph = Graph()
for i in range(N):
    for j in range(N):
        graph.add_edge(i,j, travel_time=M[i,j])

# Parameters of the problem

source = 0
destination = 9

aco = ACO(graph)
dijkstra = Dijkstra(graph)

dijkstra_path, dijkstra_cost = dijkstra.find_shortest_path(source, destination)
aco_path, aco_cost = aco.find_shortest_path(source, destination)

# Results
print(f"ACO - path: {aco_path}, cost: {aco_cost}")
print(f"Dijkstra - path: {dijkstra_path}, cost: {dijkstra_cost}")