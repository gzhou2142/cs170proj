import solver_cython
import math
import networkx as nx
import random
import solver_cython

class simulated_annealing(object):
    def __init__(self, locations, homes, start, shortest,   T = -1, alpha = -1, stopping_T = -1, stopping_iter = -1):
        self.N = len(homes)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-11 if stopping_T == -1 else stopping_T
        self.stopping_iter = 50000 if stopping_iter == -1 else stopping_iter

        self.iteration = 1
        self.shortest = shortest

        self.homes = homes
        self.start = start
        self.locations = locations

        self.best_solution = None
        self.best_fitness = float('inf')


    def initial_solution(self):

        solution = solver_cython.three_opt(solver_cython.fast_nearest_neighbor_tour(self.homes, self.start, self.shortest), self.shortest)
        start_index = solution.index(self.start)
        solution = solution[start_index:] + solution[:start_index]
        cur_fit = self.fitness(solution)
        return solution, cur_fit

    def fitness(self,solution):
        walk_cost = solver_cython.calc_walking_cost(solver_cython.find_drop_off_mapping(solution, self.homes, self.shortest), self.shortest)
        cur_fit = 0
        drive_cost = solver_cython.calc_driving_cost(solution + [self.start], self.shortest)
        return drive_cost + walk_cost
    
    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate
    
    def anneal(self):
        self.cur_solution, self.cur_fitness = self.initial_solution()

 
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            #break
            coin_flip = random.random()

            if coin_flip <= 0.5:

                candidate = set(self.cur_solution)
                remove_element = random.sample(self.cur_solution, 1)
                candidate.remove(remove_element[0])
                tour = solver_cython.fast_nearest_neighbor_tour(list(candidate), self.start, self.shortest)
                candidate = solver_cython.three_opt(tour ,self.shortest)
                start_index = candidate.index(self.start)
                candidate = candidate[start_index:] + candidate[:start_index]
                self.accept(candidate)
            else:

                candidate = set(self.cur_solution)
                remain = set(self.locations) - candidate
                add_element = random.sample(remain, 1)
                candidate.add(add_element[0])
                tour = solver_cython.fast_nearest_neighbor_tour(list(candidate), self.start, self.shortest)
                candidate = solver_cython.three_opt(tour ,self.shortest)
                start_index = candidate.index(self.start)
                candidate = candidate[start_index:] + candidate[:start_index]
                self.accept(candidate)
            print(self.cur_fitness, self.cur_solution)
            self.T *= self.alpha
            self.iteration += 1
        #self.best_solution, self.best_fitness = self.cur_solution, self.cur_fitness
        



    def batch_anneal(self, times = 10):
        for i in range(1, times + 1):
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

        return
    
    def get_cost(self):
        return self.best_fitness
    def get_solution(self):
        solution = solver_cython.three_opt(solver_cython.fast_nearest_neighbor_tour(self.best_solution, self.start, self.shortest), self.shortest)
        return solution