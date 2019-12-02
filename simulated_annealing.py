import solver_cython
import math
import networkx as nx
import random

class simulated_annealing(object):
    def __init__(self, homes, shortest,   T = -1, alpha = -1, stopping_T = -1, stopping_iter = -1):
        self.N = len(homes)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T
        self.alpha = 0.999 if alpha == -1 else alpha
        self.stopping_temperature = 1e-11 if stopping_T == -1 else stopping_T
        self.stopping_iter = 500000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.shortest = shortest
        #self.homes = homes
        self.nodes = homes

        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_list  = []

    def initial_solution(self):
        cur_node = random.choice(self.nodes)
        solution = [cur_node]
        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.shortest[cur_node][x])  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def fitness(self,solution):
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.shortest[solution[i % self.N]][solution[(i+1)%self.N]]
        return (2/3)*cur_fit
    
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

        #print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        #print("Best fitness obtained: ", self.best_fitness)
        #improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        #print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times = 10):
        for i in range(1, times + 1):
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()
            print(self.best_fitness)
        return
    
    def get_cost(self):
        return self.best_fitness
    def get_solution(self):
        return self.best_solution