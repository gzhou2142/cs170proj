
import os
import sys
sys.path.append('..')
sys.path.append('../..')
#sys.path.append('C:\\program files (x86)\\python\\lib\\site-packages')
import argparse
import utils
import numpy as np
import networkx as nx
from student_utils import *
import student_utils
import acopy as aco
import itertools

"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    if params[0] == 'naive':
        return naive_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'greedy':
        return greedy_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'three_opt':
        return three_opt_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'ant_colony':
        return ant_colony(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'greedy_clustering_three_opt':
        return greedy_clustering_three_opt(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'mst':
        return mst_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    else:
        pass

"""
makes everyone walk back home.
""" 
def naive_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    car_path = [int(starting_car_location)]
    drop_off = {int(starting_car_location): [int(h) for h in list_of_homes]}
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/naive.log', [cost], separator = '\n', append = True)
    return car_path, drop_off
"""
drop of everyone at their homes
"""
def greedy_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    car_path, visit_order = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/greedy.log', [cost], separator = '\n', append = True)
    return car_path, drop_off

"""
Uses the three opt heuristic to calculate a path that sends everyone home with no walking required.
"""
def three_opt_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    _, visit_order = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)
    visit_order = three_opt(visit_order, all_pairs_shortest_path)
    car_path = generate_full_path(visit_order, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/three_opt.log', [cost], separator = '\n', append = True)
    print(len(list_of_locations),'locations', 'three_opt:', cost)
    #print(car_path)
    return car_path, drop_off

def mst_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    verticies = set(list_of_homes)
    verticies.add(int(starting_car_location))
    verticies = list(verticies)
    newGraph = build_tour_graph(G, verticies, all_pairs_shortest_path)
    mst = nx.minimum_spanning_tree(newGraph)
    mst_tour = list(nx.dfs_preorder_nodes(newGraph, source = int(starting_car_location)))
    mst_tour.append(int(starting_car_location))
    car_path = generate_full_path(mst_tour,G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    return car_path, drop_off
    


def greedy_clustering_three_opt(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    def findsubsets(s,n):
        result = []
        for i in range(n):
            ls = [list(x) for x in list(itertools.combinations(s, i + 1))]
            result.extend(ls)
        return result
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    shortest = dict(nx.floyd_warshall(G))
    tour = [int(starting_car_location)]
    stops = [int(starting_car_location)]
    remain_bus_stop = set([int(l) for l in list_of_locations])
    remain_bus_stop.remove(int(starting_car_location))
    
    drop_off_map = find_drop_off_mapping(tour, list_of_homes, shortest)
    min_walk_cost = calc_walking_cost(drop_off_map, shortest) 
    min_drive_cost =  calc_driving_cost(tour, shortest)
    minCost = min_walk_cost + min_drive_cost
    while True:
        bestTour = None
        bestStop = None
        bestCost = minCost
        bstops = findsubsets(remain_bus_stop, 2)
        for bstop in bstops:
            new_tour = stops + bstop
            new_drop_off_map = find_drop_off_mapping(new_tour, list_of_homes, shortest)
            #new_graph = build_tour_graph(G, new_tour, shortest)
            _, new_tour = nearest_neighbor_tour(new_tour, starting_car_location, shortest, G)
            new_tour = three_opt(new_tour, shortest)
            #new_car_path = generate_full_path(new_tour, G)
            new_walk_cost = calc_walking_cost(new_drop_off_map, shortest)
            new_drive_cost = calc_driving_cost(new_tour, shortest)
            new_cost = new_walk_cost + new_drive_cost
            if new_cost < bestCost:
                bestStop = bstop
                bestCost = new_cost
                bestTour = new_tour
            sys.stdout.write(str(bstop) + '\n')  
            sys.stdout.flush()
        
        if bestCost < minCost:
            for b in bestStop:
                remain_bus_stop.remove(int(b))
            minCost = bestCost
            tour = bestTour
            stops = stops + bestStop
            #print(minCost)
            sys.stdout.write(str(minCost) + '\n')  # same as print
            sys.stdout.flush()
        else:
            break
    car_path = generate_full_path(tour, G)
    drop_off = find_drop_off_mapping(tour, list_of_homes, shortest)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/greedy_clustering_three_opt.log', [cost], separator = '\n', append = True)
    print(len(list_of_locations),'locations', 'greedy_clustering_three_opt:', cost)
    return car_path, drop_off
            
def greedy_clustering_three_opt_best_ratio(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    def findsubsets(s,n):
        result = []
        for i in range(n):
            ls = [list(x) for x in list(itertools.combinations(s, i + 1))]
            result.extend(ls)
        return result
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    shortest = dict(nx.floyd_warshall(G))
    tour = [int(starting_car_location)]
    stops = [int(starting_car_location)]
    remain_bus_stop = set([int(l) for l in list_of_locations])
    remain_bus_stop.remove(int(starting_car_location))
    
    drop_off_map = find_drop_off_mapping(tour, list_of_homes, shortest)
    walk_cost = calc_walking_cost(drop_off_map, shortest) 
    drive_cost =  calc_driving_cost(tour, shortest)
    bestCost = walk_cost + drive_cost
    while True:
        bestTour = None
        bestStop = None
        best_ratio = 0
        bstops = findsubsets(remain_bus_stop, 3)
        for bstop in bstops:
            new_tour = stops + bstop
            new_drop_off_map = find_drop_off_mapping(new_tour, list_of_homes, shortest)
            _, new_tour = nearest_neighbor_tour(new_tour, starting_car_location, shortest, G)
            new_tour = three_opt(new_tour, shortest)
            new_walk_cost = calc_walking_cost(new_drop_off_map, shortest)
            walk_cost_decrease = walk_cost - new_walk_cost
            new_drive_cost = calc_driving_cost(new_tour, shortest)
            drive_cost_increase =  new_drive_cost - drive_cost
            #print(walk_cost_decrease, drive_cost_increase)
            if drive_cost_increase > 0 and  best_ratio < (walk_cost_decrease/drive_cost_increase):
                bestStop = bstop
                best_ratio = walk_cost_decrease/drive_cost_increase
                bestTour = new_tour
                walk_cost = new_walk_cost
                drive_cost = new_drive_cost
                bestCost = new_walk_cost + new_drive_cost
        
        if best_ratio > 0:
            for b in bestStop:
                remain_bus_stop.remove(int(b))
            tour = bestTour
            stops = stops + bestStop
            sys.stdout.write(str(bestCost) + '\n')  # same as print
            sys.stdout.flush()
        else:
            break
    car_path = generate_full_path(tour, G)
    drop_off = find_drop_off_mapping(tour, list_of_homes, shortest)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/greedy_clustering_three_opt.log', [cost], separator = '\n', append = True)
    print(len(list_of_locations),'locations', 'greedy_clustering_three_opt:', cost)
    return car_path, drop_off            


def christofides(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    pass

def ant_colony(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    _, tour = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G) 
    tour = tour[1:]
    newGraph = build_tour_graph(G, tour, all_pairs_shortest_path)
    solution = ant_colony_tour(newGraph, starting_car_location)
    car_path = generate_full_path(solution, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/ant_colony.log', [cost], separator = '\n', append = True)
    print(len(list_of_locations),'locations', 'ant_colony:', cost)
    return car_path, drop_off
    

"""
finds a tour greedily
Input:
    locations: a list of locations to visit
    starting_car_location: starting location for car
    adjacency_matrix: graph representation
Output:
    A list that contains the visited locations in order
""" 
def nearest_neighbor_tour(locations, starting_car_location, all_pairs_shortest_path, G):
    if len(locations) == 1:
        return [starting_car_location]
    shortest = all_pairs_shortest_path
    set_of_locations = set(locations)
    set_of_locations.add(starting_car_location)
    tour = [int(starting_car_location)]
    visitOrder = [int(starting_car_location)]
    set_of_locations.remove(starting_car_location)
    while len(set_of_locations) > 0:
        current_node = tour.pop()
        closestLen = float('inf')
        closestNode = None
        for n in set_of_locations:
            if shortest[int(current_node)][int(n)] < closestLen:
                closestLen = shortest[int(current_node)][int(n)]
                closestNode = n
        shortestLocalPath = nx.shortest_path(G, source = int(current_node), target = int(closestNode), weight = 'weight')
        tour.extend(shortestLocalPath)
        visitOrder.append(int(closestNode))
        set_of_locations.remove(closestNode)
    tour.extend(nx.shortest_path(G, source = int(tour.pop()), target = int(starting_car_location), weight = 'weight'))
    visitOrder.append(int(starting_car_location))
    return tour, visitOrder

"""
returns optimal drop off mapping
Input:
    tour: a list that contains the tour
    list_of_homes: list of homes
    all_pairs_shortest_path: shortest paths between all pairs of vertices
Output:
    an optimal drop off mapping of homes to the vertices visited in the tour
""" 
def find_drop_off_mapping(tour, list_of_homes, all_pairs_shortest_path):
    drop_off_mapping = dict()
    shortest = all_pairs_shortest_path
    for home in list_of_homes:
        minLoc = None
        minDist = float('inf')
        for t in tour:
            if shortest[int(home)][int(t)] < minDist:
                minDist = shortest[int(home)][int(t)]
                minLoc = t
        if int(minLoc) in drop_off_mapping:
            temp = drop_off_mapping[int(minLoc)]
            temp.append(int(home))
            drop_off_mapping[int(minLoc)] = temp

        else:
            drop_off_mapping[int(minLoc)] = [int(home)]
    return drop_off_mapping


"""
calculates biggest gain given a 3 edge swap
"""
def calculateGain(tour, i, j, k, shortest):
    A,B,C,D,E,F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k]
    d0 = shortest[A][B] + shortest[C][D] + shortest[E][F]
    d1 = shortest[A][B] + shortest[C][E] + shortest[D][F]
    d2 = shortest[A][C] + shortest[B][D] + shortest[E][F]
    d3 = shortest[A][C] + shortest[B][E] + shortest[D][F]
    d4 = shortest[A][D] + shortest[E][B] + shortest[C][F]
    d5 = shortest[A][D] + shortest[E][C] + shortest[B][F]
    d6 = shortest[A][E] + shortest[D][B] + shortest[C][F]
    d7 = shortest[A][E] + shortest[D][C] + shortest[B][F]

    
    swapList = [(d0, 0), (d1, 1), (d2, 2), (d3, 3), (d4, 4), (d5, 5), (d6, 6),(d7, 7)]
    minSwap = min(swapList, key = lambda x: x[0])
    if minSwap[1] - d0 == 0:
        return (0, 0)

    return (-d0 + minSwap[0], minSwap[1])

"""
performs the 3 edge swap
"""
def move3(tour, i, j, k, case):
    if case == 1:
        tour[j:k] = reversed(tour[j:k])
    elif case == 2:
        tour[i:j] = reversed(tour[i:j])
    elif case == 3:
        tour[i:j] = reversed(tour[i:j])
        tour[j:k] = reversed(tour[j:k])
    elif case == 4:
        tour = tour[:i] + tour[j:k] + tour[i:j] + tour[k:]
    elif case == 5:
        tour = tour[:i] + tour[j:k] + list(reversed(tour[i:j])) + tour[k:]
    elif case == 6:
        tour = tour[:i] + list(reversed(tour[j:k])) + tour[i:j] + tour[k:]
    elif case == 7:
        tour = tour[:i] + list(reversed(tour[j:k])) + list(reversed(tour[i:j])) + tour[k:]
    return tour
"""
generates a list that contain all possible 3 edge combinations
"""
def all_segments(tour):
    segments = []
    for i in range(1,len(tour) - 2):
        for j in range(i+2, len(tour) - 1):
            for k in range(j+2, len(tour)):
                segments.append((i,j,k))
    return segments
"""
best improving three opt
"""
def three_opt(tour, shortest):
    while True:
        bestMove = None
        bestGain = 0
        bestCase = 0
        for (i,j,k) in all_segments(tour):
            currentGain, currentCase = calculateGain(tour, i, j, k, shortest)
            if bestGain > currentGain:
                bestGain = currentGain
                bestCase = currentCase
                bestMove = (i,j,k)
        if bestGain >= -0.00001:
            break
        else:
            i,j,k = bestMove
            tour = move3(tour, i, j, k, bestCase)
    return tour
"""
first improving three opt
"""
def three_opt_first(tour, shortest):
    while True:
        bestGain = 0
        for (i,j,k) in all_segments(tour):
            currentGain, currentCase = calculateGain(tour, i, j, k, shortest)
            if currentGain < 0:
                tour = move3(tour, i, j, k, currentCase)
                bestGain = currentGain
                break
        if bestGain >= -0.00001:
            break
    return tour

"""
generates a full path
"""
def generate_full_path(tour, G):
    final_tour = [tour[0]]
    tour[1:]
    for t in tour:
        current = final_tour.pop()
        shortestLocalPath = nx.shortest_path(G, source = int(current), target = int(t), weight = 'weight')
        final_tour.extend(shortestLocalPath)
    return final_tour

"""
builds a graph using only verticies in the list TOUR.
"""
def build_tour_graph(G, tour, all_pairs_shortest_path):
    shortest = all_pairs_shortest_path
    newGraph = nx.Graph()
    for t in tour:
        newGraph.add_node(int(t))
    for i in range(len(tour)):
        for j in range(i, len(tour)):
            if int(tour[i]) == int(tour[j]):
                continue
            else:
                newGraph.add_edge(int(tour[i]), int(tour[j]), weight = shortest[int(tour[i])][int(tour[j])])
    return newGraph

"""
finds TSP tour using ant colony optimization
"""
def ant_colony_tour(G, start):
    solver = aco.Solver(rho=0.01, q = 1)
    colony = aco.Colony(alpha = 1, beta = 5)
    tour = solver.solve(G, colony, limit = 500, gen_size = 1000)
    tour_list = tour.nodes
    start_index = tour_list.index(int(start))
    tour_list = tour_list[start_index:] + tour_list [:start_index] + [int(start)]
    return tour_list

"""
finds cost of walking given drop off mapping and all pairs shortest path
"""
def calc_walking_cost(dropoff_mapping, all_pair_shortest):
    walking_cost = 0
    dropoffs = dropoff_mapping.keys()
    for drop_location in dropoffs:
        for house in dropoff_mapping[drop_location]:
            walking_cost += all_pair_shortest[drop_location][house]
    return walking_cost

def calc_driving_cost(tour, all_pairs_shortest):
    driving_cost = 0
    if len(tour) == 1:
        return 0
    else:
        driving_cost = 0
        for i in range(1, len(tour)):
            driving_cost += all_pairs_shortest[tour[i-1]][tour[i]]
        return (2/3) * driving_cost
"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'
    
    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)
    
    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename, "")
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    if params[0] == 'naive':
        print("Using naive method")
        print("Clearing logs")
        utils.clear_file('logs/naive.log')
    elif params[0] == 'greedy':
        print('Using greedy method')
        print("Clearning logs")
        utils.clear_file('logs/greedy.log')
    elif params[0] == 'three_opt':
        print('Using three_opt method')
        print("Clearning logs")
        utils.clear_file('logs/three_opt.log')
    elif params[0] == 'ant_colony':
        print("Using ant colony optimization")
        print("Clearing logs")
        utils.clear_file("logs/ant_colony.log")
    elif params[0] == 'greedy_clustering_three_opt':
        print("Using greedy clustering three opt")
        print("Clearing logs")
        utils.clear_file("logs/greedy_clustering_three_opt.log")
    elif params[0] == 'mst':
        print("Using mst method")
        print("Clearing logs")
        utils.clear_file("logs/mst.log")
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)
    print()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)


        
