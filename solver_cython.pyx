
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
import time

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
    locations = student_utils.convert_locations_to_indices(list_of_locations, list_of_locations)
    homes = student_utils.convert_locations_to_indices(list_of_homes, list_of_locations)
    start = list_of_locations.index(starting_car_location)

    start_time = time.time()



    if params[0] == 'naive':
        car_path, drop_off = naive_solver(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'greedy':
        car_path, drop_off = greedy_solver(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'three_opt':
        car_path, drop_off = three_opt_solver(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'ant_colony':
        car_path, drop_off = ant_colony(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'greedy_clustering_three_opt':
        car_path, drop_off = greedy_clustering_three_opt(locations, homes, start, adjacency_matrix, int(params[1]))
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'mst':
        car_path, drop_off = mst_solver(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'two_opt':
        car_path, drop_off = two_opt_solver(locations, homes, start, adjacency_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'greedy_clustering_two_opt':
        car_path, drop_off = greedy_clustering_two_opt(locations, homes, start, adjacency_matrix, int(params[1]))
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
    elif params[0] == 'remove_swap':
        car_path,drop_off = remove_swap(locations, homes, start, adjacency_matrix, int(params[1]))
        print("--- %s seconds ---" % (time.time() - start_time))
        return car_path, drop_off
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
    print(len(list_of_locations),'locations', 'greedy:', cost)
    return car_path, drop_off

"""
Uses the three opt heuristic to calculate a path that sends everyone home with no walking required.
"""
def three_opt_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    _, visit_order = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)
    visit_order = three_opt(visit_order, all_pairs_shortest_path)
    #visit_order.append(starting_car_location)
    start_index = visit_order.index(starting_car_location)
    visit_order = visit_order[start_index:] + visit_order[:start_index] + [starting_car_location]
    car_path = generate_full_path(visit_order, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/three_opt.log', [cost], separator = '\n', append = True)
    #print(len(list_of_locations),'locations', 'three_opt:', cost)
    return car_path, drop_off

def two_opt_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    _, visit_order = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)

    visit_order = two_opt(visit_order, all_pairs_shortest_path)
    start_index = visit_order.index(starting_car_location)
    visit_order = visit_order[start_index:] + visit_order[:start_index] + [starting_car_location]

    car_path = generate_full_path(visit_order, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    #print(len(list_of_locations),'locations', 'two_opt:', cost)
    return car_path, drop_off
"""
uses mst to approximate
"""
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
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    utils.write_data_to_file('logs/mst.log', [cost], separator = '\n', append = True)
    print(len(list_of_locations),'locations', 'mst:', cost)
    return car_path, drop_off



def findsubsets(s,n):
        result = []
        for i in range(n):
            ls = [list(x) for x in list(itertools.combinations(s, i + 1))]
            result.extend(ls)
        return result


"""
Greedy clustering method with local search. Uses absolute overall improvement
"""
def greedy_clustering_three_opt(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, bus_stop_look_ahead):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    starting_car_location = int(starting_car_location)
    shortest = dict(nx.floyd_warshall(G))
    cdef list tour = [starting_car_location]
    remain_bus_stop = set([int(l) for l in list_of_locations])
    remain_bus_stop.remove(int(starting_car_location))
    drop_off_map = find_drop_off_mapping(tour, list_of_homes, shortest)
    #min_walk_cost = calc_walking_cost(drop_off_map, shortest)
    #min_drive_cost =  calc_driving_cost(tour, shortest)
    #minCost = min_walk_cost + min_drive_cost
    cdef double minCost = calc_walking_cost(drop_off_map, shortest) + calc_driving_cost(tour, shortest)
    cdef loop = 1
    while loop:
        bestTour = None
        bestStop = None
        bestCost = minCost
        bstops = findsubsets(remain_bus_stop, bus_stop_look_ahead)
        for bstop in bstops:
            new_tour = tour + bstop
            new_tour = fast_nearest_neighbor_tour(new_tour, starting_car_location,shortest)
            new_tour = three_opt(new_tour, shortest)
            start_index = new_tour.index(starting_car_location)
            new_tour = new_tour[start_index:] + new_tour[:start_index]
            t_tour = new_tour + [starting_car_location]
            #need to generate full graph for drop off calculation
            full_path = generate_full_path(t_tour, G)
            new_drop_off_map = find_drop_off_mapping(full_path, list_of_homes, shortest)
            new_walk_cost = calc_walking_cost(new_drop_off_map, shortest)
            new_drive_cost = calc_driving_cost(t_tour, shortest)
            new_cost = new_walk_cost + new_drive_cost
            if new_cost < bestCost:
                bestStop = bstop
                bestCost = new_cost
                bestTour = new_tour
            #print(tour)
        if bestCost < minCost:
            for b in bestStop:
                remain_bus_stop.remove(int(b))
            minCost = bestCost
            tour = bestTour
            # sys.stdout.write(str(minCost) + '\n')  # same as print
            # sys.stdout.flush()
        else:
            loop = 0
    tour = tour + [starting_car_location]
    car_path = generate_full_path(tour, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, shortest)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    #utils.write_data_to_file('logs/greedy_clustering_three_opt.log', [cost], separator = '\n', append = True)
    #print(len(list_of_locations),'locations', 'greedy_clustering_three_opt:', cost)
    return car_path, drop_off

def rotate_to_start(tour, starting_car_location):
    start_index = tour.index(starting_car_location)
    return tour[start_index:] + tour[:start_index]

def find_k_closest(k, start, list, shortest):
    distance_dict = {}
    for l in list:
        distance_dict[l] = shortest[start][l]
    sorted_dis = sorted(distance_dict.items(), key = lambda kv: kv[1])
    sorted_dis = sorted_dis[1:k+1]
    sorted_nodes = [x[0] for x in sorted_dis]
    return sorted_nodes

def remove_swap(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, bus_stop_look_ahead):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    shortest = dict(nx.floyd_warshall(G))

    set_homes = set(list_of_homes)
    set_homes.add(starting_car_location)

    tour = list(set_homes)
    remain_bus_stop = set(list_of_locations) - set_homes
    tour = fast_nearest_neighbor_path(tour, starting_car_location, shortest)
    tour = three_opt(tour, shortest)
    tour = rotate_to_start(tour, starting_car_location)


    t_tour = tour + [starting_car_location]
    drop_off_map = find_drop_off_mapping(generate_full_path(t_tour, G), list_of_homes, shortest)

    cdef double best_cost = calc_walking_cost(drop_off_map, shortest) + calc_driving_cost(t_tour, shortest)
    best_tour = tour
    best_drop_off_map = drop_off_map
    cdef int loop = 1
    while loop:
        loop = 0
        local_cost = best_cost
        local_tour = best_tour
        for t in best_tour:
            if t == starting_car_location:
                continue
            #remove
            new_set_homes = set_homes - {t}
            new_removal_tour = fast_nearest_neighbor_path(list(new_set_homes), starting_car_location, shortest)
            new_removal_tour = two_opt(new_removal_tour, shortest)
            new_removal_tour = rotate_to_start(new_removal_tour, starting_car_location)
            t_tour = new_removal_tour + [starting_car_location]
            new_removal_full_path = generate_full_path(t_tour, G)
            new_drop_off_map = find_drop_off_mapping(new_removal_full_path, list_of_homes, shortest)
            removal_cost =  calc_driving_cost(t_tour, shortest) + calc_walking_cost(new_drop_off_map, shortest)
            #print(t)
            if removal_cost < local_cost:

                local_cost = removal_cost
                #print(local_cost)
                local_tour = new_removal_tour
                loop = 1
            #add neighbors
            remain_bus_stop.add(t)
            k_closest = find_k_closest(bus_stop_look_ahead, t, remain_bus_stop, shortest)
            k_closest_subsets = findsubsets(k_closest, len(k_closest))
            remain_bus_stop.remove(t)
            #print(t)
            #print(k_closest)
            for s in k_closest_subsets:
                add_set_homes = new_set_homes | set(s)
                new_add_tour = fast_nearest_neighbor_path(list(add_set_homes), starting_car_location, shortest)
                new_add_tour = two_opt(new_add_tour, shortest)
                new_add_tour = rotate_to_start(new_add_tour, starting_car_location)
                t_tour = new_add_tour + [starting_car_location]
                new_add_full_path = generate_full_path(t_tour, G)
                new_drop_off_map = find_drop_off_mapping(new_add_full_path, list_of_homes, shortest)
                add_cost = calc_driving_cost(t_tour, shortest) + calc_walking_cost(new_drop_off_map, shortest)
                if add_cost < local_cost:
                    local_cost = add_cost
                    #print(local_cost)
                    local_tour = new_add_tour
                    loop = 1
        #print(best_cost)
        best_cost = local_cost
        #print(best_cost)
        best_tour = local_tour
        #print(best_tour)
        #print(best_cost)


    best_tour = best_tour + [starting_car_location]
    full_best_tour = generate_full_path(best_tour, G)
    best_drop_off_map = find_drop_off_mapping(full_best_tour, list_of_homes, shortest)
    return full_best_tour, best_drop_off_map




"""
Greedy clustering using two opt local seearch.
"""
def greedy_clustering_two_opt(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, bus_stop_look_ahead):

    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    starting_car_location = int(starting_car_location)
    shortest = dict(nx.floyd_warshall(G))
    cdef list tour = [starting_car_location]
    remain_bus_stop = set([int(l) for l in list_of_locations])
    remain_bus_stop.remove(int(starting_car_location))
    drop_off_map = find_drop_off_mapping(tour, list_of_homes, shortest)
    #min_walk_cost = calc_walking_cost(drop_off_map, shortest)
    #min_drive_cost =  calc_driving_cost(tour, shortest)
    #minCost = min_walk_cost + min_drive_cost
    cdef double minCost = calc_walking_cost(drop_off_map, shortest) + calc_driving_cost(tour, shortest)
    cdef loop = 1
    while loop:
        bestTour = None
        bestStop = None
        bestCost = minCost
        bstops = findsubsets(remain_bus_stop, bus_stop_look_ahead)
        for bstop in bstops:
            new_tour = tour + bstop
            new_tour = fast_nearest_neighbor_tour(new_tour, starting_car_location,shortest)
            new_tour = three_opt(new_tour, shortest)
            start_index = new_tour.index(starting_car_location)
            new_tour = new_tour[start_index:] + new_tour[:start_index]
            t_tour = new_tour + [starting_car_location]
            #need to generate full graph for drop off calculation
            #full_path = generate_full_path(t_tour, G)
            new_drop_off_map = find_drop_off_mapping(t_tour, list_of_homes, shortest)
            new_walk_cost = calc_walking_cost(new_drop_off_map, shortest)
            new_drive_cost = calc_driving_cost(t_tour, shortest)
            new_cost = new_walk_cost + new_drive_cost
            if new_cost < bestCost:
                bestStop = bstop
                bestCost = new_cost
                bestTour = new_tour
            #print(tour)
        #print(minCost)
        if bestCost < minCost:
            for b in bestStop:
                remain_bus_stop.remove(int(b))
            minCost = bestCost
            tour = bestTour
            # sys.stdout.write(str(minCost) + '\n')  # same as print
            # sys.stdout.flush()
        else:
            loop = 0

    tour = three_opt(tour, shortest)
    tour = tour + [starting_car_location]
    car_path = generate_full_path(tour, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, shortest)
    cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    #utils.write_data_to_file('logs/greedy_clustering_three_opt.log', [cost], separator = '\n', append = True)
    #print(len(list_of_locations),'locations', 'greedy_clustering_three_opt:', cost)
    return car_path, drop_off

    



def ant_colony(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    _, tour = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)
    #tour = tour[1:]

    newGraph = build_tour_graph(G, tour, all_pairs_shortest_path)
    solution = ant_colony_tour(newGraph, starting_car_location)
    car_path = generate_full_path(solution, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
    #cost, _ = student_utils.cost_of_solution(G, car_path, drop_off)
    #utils.write_data_to_file('logs/ant_colony.log', [cost], separator = '\n', append = True)
    #print(len(list_of_locations),'locations', 'ant_colony:', cost)
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
    #visitOrder.append(int(starting_car_location))
    return tour, visitOrder
"""
finds a tour using nearest neighbor greedy algorithm. this is the same algorithm as above except it is optimized for the greedy_clustering_three_opt algorithm
"""
def fast_nearest_neighbor_tour(locations, starting_car_locations, shortest):
    if len(locations) == 1:
        return [starting_car_locations]
    cdef set set_of_locations = set(locations)
    set_of_locations.add(starting_car_locations)
    cdef list tour = [starting_car_locations]
    set_of_locations.remove(starting_car_locations)
    cdef int remaining_locations = len(set_of_locations)
    cdef int current_node = tour[-1]
    while remaining_locations > 0:
        current_node = tour[-1]
        closestLen = float('inf')
        closestNode = None
        for n in set_of_locations:
            if shortest[int(current_node)][int(n)] < closestLen:
                closestLen = shortest[int(current_node)][int(n)]
                closestNode = int(n)
        tour.append(closestNode)
        set_of_locations.remove(closestNode)
        remaining_locations -= 1
    #tour.append(starting_car_locations)

    return tour

def fast_nearest_neighbor_path(locations, starting_car_locations, shortest):
    if len(locations) == 1:
        return [starting_car_locations]
    cdef set set_of_locations = set(locations)
    set_of_locations.add(starting_car_locations)
    cdef list tour = [starting_car_locations]
    set_of_locations.remove(starting_car_locations)
    cdef int remaining_locations = len(set_of_locations)
    cdef int current_node = tour[-1]
    while remaining_locations > 0:
        current_node = tour[-1]
        closestLen = float('inf')
        closestNode = None
        for n in set_of_locations:
            if shortest[int(current_node)][int(n)] < closestLen:
                closestLen = shortest[int(current_node)][int(n)]
                closestNode = int(n)
        tour.append(closestNode)
        set_of_locations.remove(closestNode)
        remaining_locations -= 1
    #tour.append(starting_car_locations)

    return tour

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
def calculateGain( A,B,C,D,E,F, shortest):
    #A,B,C,D,E,F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k]
    # cdef int A = tour[i-1]
    # cdef int B = tour[i]
    # cdef int C = tour[j-1]
    # cdef int D = tour[j]
    # cdef int E = tour[k-1]
    # cdef int F = tour[k]
    cdef double AB = shortest[A][B]
    cdef double CD = shortest[C][D]
    cdef double EF = shortest[E][F]
    cdef double CE = shortest[C][E]
    cdef double DF = shortest[D][F]
    cdef double AC = shortest[A][C]
    cdef double BD = shortest[B][D]
    cdef double BE = shortest[B][E]
    cdef double AD = shortest[A][D]
    cdef double CF = shortest[C][F]
    cdef double BF = shortest[B][F]
    cdef double AE = shortest[A][E]


    #cdef double CE, DF, AC, BD, BE = shortest[C][E], shortest[D][F], shortest[A][C], shortest[B][D], shortest[B][E]
    #cdef double AD, CF, BF, AE = shortest[A][D], shortest[C][F], shortest[B][F], shortest[A][E]
    cdef double d0 = AB + CD + EF
    cdef double d1 = AB + CE + DF
    cdef double d2 = AC + BD + EF
    cdef double d3 = AC + BE + DF
    cdef double d4 = AD + BE + CF
    cdef double d5 = AD + CE + BF
    cdef double d6 = AE + BD + CF
    cdef double d7 = AE + CD + BF


    #swapList = [(d0, 0), (d1, 1), (d2, 2), (d3, 3), (d4, 4), (d5, 5), (d6, 6),(d7, 7)]
    swapList = [(d0, 0), (d1, 1), (d2, 2), (d3, 3), (d4, 4), (d5, 5), (d6, 6),(d7, 7)]
    minSwap = min(swapList)
    gain = minSwap[0] - d0
    if gain == 0:
        return (0, 0)

    return (gain, minSwap[1])


"""
performs the 3 edge swap
"""
def move3(tour, i, j, k, case):
    #i = i -1
    #j = j - 1
    #k = k - 1
    #print(i,j,k)
    N = len(tour)
    if case == 1:
        #tour[j:k] = reversed(tour[j:k])
        tour = reverse_segment(tour, (k+1)%N, i)
    elif case == 2:
        #tour[i:j] = reversed(tour[i:j])
        #tour[(j+1)%N :k] = reversed(tour[(j+1)%N :k])
        tour = reverse_segment(tour, (j+1) % N, k)
    elif case == 3:
        #tour[i:j] = reversed(tour[i:j])
        #tour[j:k] = reversed(tour[j:k])
        #tour[(j+1)%N : k] = reversed(tour[(j+1)%N : k])
        tour = reverse_segment(tour, (i+1)%N, j)
    elif case == 4:
        #tour = tour[:i] + tour[j:k] + tour[i:j] + tour[k:]
        tour = reverse_segment(tour, (j+1)%N, k)
        tour = reverse_segment(tour, (i+1)%N, j)
    elif case == 5:
        #tour = tour[:i] + tour[j:k] + list(reversed(tour[i:j])) + tour[k:]
        tour = reverse_segment(tour, (k+1)%N, i)
        tour = reverse_segment(tour, (i+1)%N, j)
    elif case == 6:
        #tour = tour[:i] + list(reversed(tour[j:k])) + tour[i:j] + tour[k:]
        tour = reverse_segment(tour, (k+1)%N, i)
        tour = reverse_segment(tour, (j+1)%N, k)
    elif case == 7:
        #tour = tour[:i] + list(reversed(tour[j:k])) + list(reversed(tour[i:j])) + tour[k:]
        tour = reverse_segment(tour, (k+1)%N, i)
        tour = reverse_segment(tour, (i+1)%N, j)
        tour = reverse_segment(tour, (j+1)%N, k)
    return tour
"""
generates a list that contain all possible 3 edge combinations
"""
# def all_segments(tour):
#     segments = []
#     for i in range(1,len(tour) - 2):
#         for j in range(i+2, len(tour) - 1):
#             for k in range(j+2, len(tour)):
#                 segments.append((i,j,k))
#     return segments

def all_segments(tour):
    #segments = []
    for i in range(1,len(tour) - 2):
        for j in range(i+2, len(tour) - 1):
            for k in range(j+2, len(tour)):
                yield (i,j,k)
    #return segments

def gain_two_opt(X1, X2, Y1, Y2, shortest):
    cdef double del_length = shortest[X1][X2] + shortest[Y1][Y2]
    cdef double add_length = shortest[X1][Y1] + shortest[X2][Y2]
    return del_length - add_length

def make_2_opt_move(tour, i, j):
    return reverse_segment(tour, (i + 1) % len(tour), j)

"""
Two opt
"""
def two_opt(tour, shortest):
    cdef int local_optimal = 0
    cdef int N = len(tour)
    while not local_optimal:
        local_optimal = 1
        best_gain = 0
        best_move = None
        for counter_1 in range(N - 2):
            i = counter_1
            X1 = tour[i]
            X2 = tour[(i+1)%N]
            counter_2_Limit = 0
            if i == 0:
                counter_2_Limit = N -1
            else:
                counter_2_Limit = N
            for counter_2 in range(i+2, counter_2_Limit):
                j = counter_2
                Y1 = tour[j]
                Y2 = tour[(j+1) %N]
                gain_expected = gain_two_opt(X1,X2,Y1,Y2, shortest)
                if gain_expected > best_gain:
                    best_gain = gain_expected
                    best_move = (i,j)
                    local_optimal = 0
        if not local_optimal:
            tour = make_2_opt_move(tour, best_move[0], best_move[1])
    return tour
# def two_opt( tour,  shortest):
#     cdef list best = tour
#     cdef int improved = 1
#     while improved:
#         improved = 0
#         gain = 0
#         bestSwap = None
#         for i in range(1, len(tour) - 2):
#             for j in range(i + 1, len(tour)):
#                 A,B,C,D = tour[i-1], tour[i], tour[j-1], tour[j]
#                 currentGain = shortest[A][C] + shortest[B][D] - (shortest[A][B] + shortest[C][D])
#                 if currentGain < gain:
#                     gain = currentGain
#                     bestSwap = (i, j)
#                     improved = 1
#         if bestSwap != None:
#             tour[bestSwap[0]:bestSwap[1]] = tour[bestSwap[1]-1:bestSwap[0]-1:-1]
#     return tour


def max_gain_from_3_opt(x1, x2, y1, y2, z1, z2, shortest):
    cdef double x1x2 = shortest[x1][x2]
    cdef double z1z2 = shortest[z1][z2]
    cdef double x1z1 = shortest[x1][z1]
    cdef double x2z2 = shortest[x2][z2]
    cdef double y1y2 = shortest[y1][y2]
    cdef double y1z1 = shortest[y1][z1]
    cdef double y2z2 = shortest[y2][z2]
    cdef double x1y1 = shortest[x1][y1]
    cdef double x2y2 = shortest[x2][y2]
    cdef double y1z2 = shortest[y1][z2]
    cdef double x1y2 = shortest[x1][y2]
    cdef double x2z1 = shortest[x2][z1]
    cdef double delLength = x1x2 + y1y2 + z1z2
    opt0 = (0,0)
    opt1 = (x1x2 + z1z2 - x1z1 - x2z2, 1)
    opt2 = (y1y2 + z1z2 - y1z1 - y2z2, 2)
    opt3 = (x1x2 + y1y2 - x1y1 - x2y2, 3)
    opt4 = (delLength - x1y1 - x2z1 - y2z2, 4)
    opt5 = (delLength - x1z1 - x2y2 - y1z2, 5)
    opt6 = (delLength - x1y2 - y1z1 - x2z2, 6)
    opt7 = (delLength - x1y2 - x2z1 - y1z2, 7)
    ls = [opt0, opt1, opt2, opt3, opt4, opt5, opt6, opt7]
    #ls = [opt0, opt3, opt6, opt7]
    val = max(ls, key = lambda x : x[0])
    if val[0] >= 0.000017:
        return val
    else:
        return (0,0)

def reverse_segment(tour, start, end):
    #return tour
    N = len(tour)
    #tour[x: y + 1] = reversed(tour[x: y + 1])
    inversionSize = int(((N + end - start + 1)%N)/2)
    left = start
    right = end
    for c in range(1, inversionSize + 1):
        tour[left],tour[right] = tour[right],tour[left]
        left = (left + 1) % N
        right = (N + right - 1) %N
    return tour

"""
best improving three opt
"""
def three_opt(tour, shortest):
    cdef int local_optimal = 0
    cdef int N = len(tour)
    while not local_optimal:
        local_optimal = 1
        best_gain = 0
        best_case = 0
        bestMove = None
        for counter_1 in xrange(N):
            i = counter_1
            X1 = tour[i]
            X2 = tour[(i+1) % N]
            for counter_2 in xrange(1, N - 2):
                j = (i + counter_2) % N
                Y1 = tour[j]
                Y2 = tour[(j+1) % N]
                for counter_3 in xrange(counter_2 + 1, N):
                    k = (i + counter_3) % N
                    Z1 = tour[k]
                    Z2 = tour[(k+1) % N]
                    currentGain, currentCase = max_gain_from_3_opt(X1, X2, Y1, Y2, Z1, Z2, shortest)
                    if currentGain > best_gain:
                        best_gain, best_case = currentGain, currentCase
                        bestMove = (i,j,k)
                        local_optimal = 0
        #print(best_gain)
        if not local_optimal:
            tour = move3(tour, bestMove[0], bestMove[1], bestMove[2], best_case)

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
    solver = aco.Solver(rho=0.1, q = 1)
    colony = aco.Colony(alpha = 1, beta = 3)
    tour = solver.solve(G, colony, limit = 500, gen_size = 1000)#, gen_size = 100)
    tour_list = tour.nodes
    start_index = tour_list.index(int(start))
    tour_list = tour_list[start_index:] + tour_list [:start_index] + [int(start)]
    return tour_list

"""
finds cost of walking given drop off mapping and all pairs shortest path
"""
def calc_walking_cost(dropoff_mapping, all_pair_shortest):
    walking_cost = 0.0
    dropoffs = dropoff_mapping.keys()
    for drop_location in dropoffs:
        for house in dropoff_mapping[drop_location]:
            walking_cost += all_pair_shortest[drop_location][house]
    return walking_cost

cdef double calc_driving_cost(tour, all_pairs_shortest):
    cdef double driving_cost = 0.0
    #cdef double ratio = 2
    if len(tour) == 1:
        return driving_cost
    else:
        driving_cost = 0
        for i in range(1, len(tour)):
            driving_cost += all_pairs_shortest[tour[i-1]][tour[i]]
        driving_cost = driving_cost * 2 / 3
        return driving_cost
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

def improve_from_file(input_file, output_directory, params = []):
    print('Processing %s' % (input_file))

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)

    new_cost, _ = student_utils.cost_of_solution(G, car_path, drop_offs)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename, "")
    output_file = f'{output_directory}/{output_filename}'
    try:
        output_data = utils.read_file(output_file)
    except:
        print("%s has cost %f" % (input_file, new_cost))
        convertToFile(car_path, drop_offs, output_file, list_locations)
        return
    car_cycle = output_data[0]
    new_drop_offs = {}
    num_dropoffs = int(output_data[1][0])
    for i in range(num_dropoffs):
        dropoff = output_data[i + 2]
        dropoff_index = list_locations.index(dropoff[0])
        new_drop_offs[dropoff_index] = convert_locations_to_indices(dropoff[1:], list_locations)
    car_cycle = student_utils.convert_locations_to_indices(car_cycle, list_locations)
    old_cost, _ = student_utils.cost_of_solution(G, car_cycle, new_drop_offs)

    if new_cost < old_cost:
        print("Improved. New cost is %f. Old cost is %f." % (new_cost, old_cost))
        #print(input_file, "improved from", old_cost, 'to', new_cost)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        convertToFile(car_path, drop_offs, output_file, list_locations)
    else:
        print("Not improved. New cost is %f. Old cost is %f." % (new_cost, old_cost))



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
