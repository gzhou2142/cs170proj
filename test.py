import solver
import networkx as nx
import solver_cython
import numpy as np
# tour = [0,1,2,3,4,5,0]
# segments = []
# for i,j,k in solver.all_segments(tour):
#     #print()
#     segments.append((i-1, i, j-1, j, k-1, k))
#print(segments)
solver.get_adjacency_matrix('inputs/233_200.in')
# g = nx.Graph()
# for i in range(6):
#     g.add_node(i)
# val = 2*(3**0.5)
# g.add_edge(0,1,weight = 2)
# g.add_edge(1,2,weight = 2)
# g.add_edge(2,3,weight = 2)
# g.add_edge(3,4,weight = 2)
# g.add_edge(4,5,weight = 2)
# g.add_edge(5,0,weight = 2)
# g.add_edge(0,3,weight=4)
# g.add_edge(1,4,weight=4)
# g.add_edge(2,5,weight=4)
# g.add_edge(0,2,weight = val)
# g.add_edge(0,4,weight = val)
# g.add_edge(1,3,weight = val)
# g.add_edge(1,5, weight = val)
# g.add_edge(2,4,weight=val)
# g.add_edge(3,5,weight=val)
# shortest = dict(nx.floyd_warshall(g))
# import matplotlib.pyplot as plt
# def cal_dist(tour):
#     dist = 0
#     for i in range(1, len(tour)):
#         #print(shortest[tour[i-1]][tour[i]])
#         dist = dist + shortest[tour[i-1]][tour[i]]
#     return dist
# spring_layout = nx.spring_layout(g, dim = 3,  weight = 'weight')
