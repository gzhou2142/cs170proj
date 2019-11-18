import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import numpy as np
import networkx as nx
from student_utils import *
import student_utils
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
        A list of (location, [homes]) representing drop-offs
    """
    if params[0] == 'naive':
        return naive_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    elif params[0] == 'greedy':
        return greedy_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    else:
        pass

"""
makes everyone walk back home.
""" 
def naive_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    car_path = [int(starting_car_location)]
    drop_off = {int(starting_car_location): [int(h) for h in list_of_homes]}
    return car_path, drop_off
"""
drop of everyone at their homes
"""
def greedy_solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    all_pairs_shortest_path = dict(nx.floyd_warshall(G))
    car_path = nearest_neighbor_tour(list_of_homes, starting_car_location, all_pairs_shortest_path, G)
    drop_off = find_drop_off_mapping(car_path, list_of_homes, all_pairs_shortest_path)
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
    tour = [int(starting_car_location)]
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
        set_of_locations.remove(closestNode)
    #tour.append(int(starting_car_location))
    tour.extend(nx.shortest_path(G, source = int(tour.pop()), target = int(starting_car_location), weight = 'weight'))
    # for i in range(len(tour)):
    #     tour[i] = int(tour[i])
    #print(tour)
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
            drop_off_mapping[int(minLoc)] = drop_off_mapping[int(minLoc)].append(int(home))
        else:
            drop_off_mapping[int(minLoc)] = [int(home)]
    return drop_off_mapping



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
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


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


        
