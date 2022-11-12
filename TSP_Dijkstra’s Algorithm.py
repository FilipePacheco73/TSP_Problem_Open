# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:52:30 2022

@author: Z52XXR7
"""

def dijkstra(current, nodes, distances):
    # These are all the nodes which have not been visited yet
    unvisited = {node: None for node in nodes}
    # It will store the shortest distance from one node to another
    visited = {}
    # It will store the predecessors of the nodes
    currentDistance = 0
    unvisited[current] = currentDistance
    # Running the loop while all the nodes have been visited
    while True:
        # iterating through all the unvisited node
        for neighbour, distance in distances[current].items():
            # Iterating through the connected nodes of current_node (for 
            # example, a is connected with b and c having values 10 and 3
            # respectively) and the weight of the edges
            if neighbour not in unvisited: continue
            newDistance = currentDistance + distance
            if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        # Till now the shortest distance between the source node and target node 
        # has been found. Set the current node as the target node
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        print(sorted(candidates, key = lambda x: x[1]))
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]
    return visited
  
nodes = ('A', 'B', 'C', 'D', 'E')
distances = {
    'A': {'A': 10, 'B': 1.18674, 'C': 1.33595, 'D': 1.58065, 'E': 1.00321},
    'B': {'A': 1.18674, 'B': 10,'C': 1.11963, 'D': 1.43422, 'E': 0.658261},
    'C': {'A': 1.33595, 'B': 1.11963, 'C': 10, 'D': 1.45493, 'E': 0.913113},
    'D': {'A': 1.58065, 'B': 1.43422, 'C': 1.45493, 'D':10, 'E': 1.10498},
    'E': {'A': 1.00321, 'B': 0.658261,'C': 0.913113,'D': 1.10498, 'E':10}}
current = 'E'
  
print(dijkstra(current, nodes, distances))
A  = (dijkstra(current, nodes, distances))
print(sum(A.values()))