import sys
sys.path.append('..')
sys.path.append('../..')
import utils
import numpy as np
import networkx as nx
import random
import itertools

# Generate a random graph with NUM_POINTS number of vertices
# SIZE_X AND SIZE_Y set the limit to the distances between verticies
def graph_generate(num_points, size_x, size_y, seed = None):
    numEdge = random.randint(num_points - 1, (num_points * (num_points- 1)/2))
    graph = nx.gnm_random_graph(num_points, (num_points * (num_points- 1)/2), seed = seed)
    random.seed(seed)
    coords = set()
    
    # for i in range(num_points):
    #     verticies.append( ((random.randint(0, size_x), random.randint(0, size_y))))
    while(len(coords) < num_points):
        x, y = random.randint(0, size_x), random.randint(0, size_y)
        while ((x,y) in coords):
            x, y = random.randint(0, size_x), random.randint(0, size_y)
        coords.add((x,y))
    #verticies = list(itertools.product(range(size_x), range(size_y)))
    verticies = list(coords)
    verticies = random.sample(verticies, num_points)
    for e in graph.edges:
        graph[e[0]][e[1]]['weight'] = distance(verticies[e[0]], verticies[e[1]])
    return graph


# Outputs distances between two points presented in the form of tuples
# e.g. distance((0,0), (3,4)) will return 5
def distance(point1, point2):
    var = round(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5, 5)
    #var1 = '%.5f' % (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5)
    
    return var

# Test function has no use
def test():
    g = graph_generate(5, 5,5)
    for (u,v,wt) in g.edges.data('weight'):
        print((u,v),wt)
    mtx = nx.adjacency_matrix(g, weight='weight')
    print(mtx.todense())

# generate a graph with NUM_LOC locations and NUM_HOMES homes and output the result to a the file FILENAME.
def generate_input(filename, num_loc, num_homes, size, seed = None):
    print('Creating:', filename)
    g = graph_generate(num_loc, size, size, seed = seed)
    locations = list(g.nodes)
    homes = random.sample(locations, num_homes)
    start = random.sample(locations, 1)
    utils.clear_file(filename)
    utils.write_to_file(filename, str(len(locations)) + '\n', append = True)
    utils.write_to_file(filename, str(len(homes)) + '\n', append = True)
    utils.append_data_next_line(filename, locations, " ", append = True)
    utils.append_data_next_line(filename, homes, " ", append = True)
    utils.append_data_next_line(filename, start, " ", append = True)
    
    matrix = nx.to_numpy_matrix(g, weight = 'weight')
    entry_len = len(str(size)) + 6
    for r in matrix:
        for c in np.nditer(r):
            if c == 0:
                utils.write_to_file(filename, 'x' + (' ' * entry_len), append = 'a')
            else:
                utils.write_to_file(filename, str(c) + (' ' * (entry_len + 1 - len(str(c)))), append = 'a')
        utils.append_next_line(filename)

#generate_input('inputs/50.in', 50, 25, 1000)
# g = graph_generate(3, 3,3)
# for u in g.edges.data('weight'):
#     print(u)