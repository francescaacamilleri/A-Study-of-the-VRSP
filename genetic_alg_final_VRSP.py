# IMPLEMENTATION OF GENETIC ALGORITHM FOR THE VRSP PROBLEM

#----------------------- IMPORT REQUIRED PACKAGES -----------------------------

import numpy as np
import pandas as pd
import copy
import random
import math
import time


start = time.time()
# -----------------------------------------------------------------------------
#----------------------- FUNCTIONS REQUIRED FOR GA ----------------------------
# -----------------------------------------------------------------------------

# This function computes the cost incurred if sol is deviating from MS
def DeviationFromMaster(MS, sol, n, F=4.0):
    solution_pairs = []
    for i in range(len(sol)-1):
        if sol[i] != n+1:
            solution_pairs.append([sol[i], sol[i+1]])
        
    add_cost = 0
    for i in range(len(MS)-1):
        pair = MS[i:i+2]
        if pair[0] == n+1:
            continue
        
        if pair not in solution_pairs:
            add_cost += F
            # print(pair, ' not in ', solution_pairs)
            
    # print(add_cost)
    
    return add_cost
        
              
# This function computes the cost incurred from implementing the solution
def SolutionCost(solution, Q, n, cost, g, MS=None):
    s = list(solution)
    new_s = [0] #This will be an array having the indices of depots where appropriate
    total_demand = 0
    for node in s:
        total_demand += d[node]
        if total_demand > Q:
            new_s.append(n+1)
            new_s.append(0)
            total_demand = 0
        new_s.append(node)
      
    new_s.append(n+1)
    
    sol_cost = 0
    for i in range(len(new_s)-1):
        sol_cost += cost[new_s[i], new_s[i+1]] 
        
    sol_cost += g * (len(new_s) - np.count_nonzero(new_s))
    
    add_cost = 0
    if MS != None:
        add_cost = DeviationFromMaster(MS, new_s, n, F=4.0)
    
    return sol_cost + add_cost

# This function is returning the best solution in the population
def BestSolution(population, Q, n, cost, g, MS=None):
    pop_size = population.shape[0]
    best_idx = 0
    best_cost = np.inf
    
    for i in range(pop_size):
        cost_current = SolutionCost(population[i], Q, n, cost, g, MS)
        if cost_current < best_cost:
            best_cost = cost_current
            best_idx = i
            
    return best_idx, best_cost, population[best_idx]

# This function is implementing the Ordered Crossover Operation
def Crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]
    
    child = childP1 + childP2
    return child

# This function is implementing the Mutation Operation
def mutate(individual, mutationRate):
    ind = copy.deepcopy(individual)
    for swapped in range(len(ind)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(ind))
            
            cust1 = ind[swapped]
            cust2 = ind[swapWith]
            
            ind[swapped] = cust2
            ind[swapWith] = cust1
    return ind

# -----------------------------------------------------------------------------
# ---------------------- IMPORT DATA/INSTANCES --------------------------------
# -----------------------------------------------------------------------------

all_distances = pd.read_excel('Main Data VRSP.xlsx', 'distanceMatrix')
all_costs = pd.read_excel('Main Data VRSP.xlsx', 'CostMatrix', header=None)
instance = pd.Series([74]).append(pd.read_excel('Main Data VRSP.xlsx', 'I1')['Node'])
instance = instance.append(pd.Series([74]))
demand = pd.Series([0]).append(pd.read_excel('Main Data VRSP.xlsx', 'I1')['Demand_VRSP'])
demand = np.asarray(demand.append(pd.Series([0])))

distance = np.zeros((instance.shape[0], instance.shape[0]))
cost = np.zeros((instance.shape[0], instance.shape[0]))
row = 0
for i in instance:
    col = 0
    for j in instance:
        distance[row][col] = all_distances.iloc[i-1][j]
        cost[row][col] = all_costs.iloc[i-1][j-1]
        col += 1
    row += 1
    
cost /= 1000.0 # in euros
Time = distance/500.0 # in minutes

# -----------------------------------------------------------------------------
# --------------------------- Master Schedules --------------------------------
# -----------------------------------------------------------------------------

# Instance 1
MS = [0,1,3,2,5,4,6]
# Instance 2
#MS = [0,3,4,6,5,1,8,2,9,0,7,9]
# Instance 3
#MS = [0, 5, 6, 1, 2, 9, 11, 13, 0, 8, 7, 4, 12, 10, 3, 13]
# Instance 4
#MS = [0, 2, 3, 6, 7, 8, 13, 21,0, 20, 14, 11, 12, 4, 19, 5, 18, 21, 0, 17, 1, 15, 10, 9, 16, 21]
# Instance 5
#MS = [0, 8, 20, 26, 14, 27, 21, 17, 12, 29, 0, 23, 11, 13, 28, 15, 2, 1, 29, 0, 9, 10, 22, 5, 4, 3, 29, 0, 6, 19, 16, 7, 18, 24, 25, 29]
# Instance 6
#MS = [0, 35, 38, 26, 28, 10, 11, 39, 0, 17, 13, 14, 7, 4, 5, 15, 39, 0, 21, 36, 29, 9, 8, 12, 22, 25, 39, 0, 30, 16, 18, 33, 6, 20, 19, 24, 39, 0, 27, 31, 32, 1, 2, 3, 23, 34, 37, 39]
# Instance 7
#MS = [0, 46, 21, 19, 18, 20, 4, 51,0, 22, 37, 48, 36, 29, 31, 23, 34, 51, 0, 32, 24, 49, 2, 1, 3, 33, 41, 40, 51, 0, 7, 5, 6, 45, 25, 16, 14, 51, 0, 17, 15, 44, 38, 50, 35, 26, 11, 51, 0, 13, 43, 12, 39, 10, 27, 30, 28, 51, 0, 42, 9, 8, 47, 51]
# Instance 8
#MS = [0, 30, 31, 35, 46, 24, 25, 71, 0, 62, 21, 26, 8, 15, 11, 14, 9, 71, 0, 13, 29, 40, 44, 39, 58, 63, 12, 33, 71, 0, 23, 53, 6, 1, 45, 5, 7, 71, 0, 2, 3, 4, 56, 41, 67, 38, 52, 68, 71, 0, 54, 10, 42, 65, 37, 59, 17, 49, 71, 0, 18, 16, 66, 36, 51, 50, 48, 43, 71, 0, 64, 34, 28, 27, 32, 22, 57, 60, 71, 0, 20, 70, 19, 61, 69, 47, 55, 71]
# Instance 9
#MS = [0, 37, 40, 38, 91, 39, 78, 100, 0, 25, 30, 55, 5, 70, 9, 67, 75, 60, 100, 0, 35, 57, 34, 36, 32, 33, 31, 100, 0, 64, 29, 26, 52, 96, 88, 42, 41, 100, 0, 71, 65, 54, 53, 61, 93, 22, 95, 100, 0, 79, 13, 20, 90, 69, 97, 58, 27, 100, 0, 17, 19, 18, 11, 14, 12, 16, 100, 0, 92, 45, 43, 47, 84, 10, 77, 85, 100, 0, 80, 23, 82, 99, 76, 94, 24, 100, 0, 51, 6, 1, 8, 68, 98, 56, 63, 83, 100, 0, 62, 73, 4, 2, 3, 86, 7, 100, 0, 48, 49, 46, 59, 44, 15, 50, 100, 0, 21, 28, 89, 72, 81, 66, 74, 87, 100]
# Cluster 1
#MS = [0, 4, 5, 6, 1, 2, 3, 13, 0, 11, 10, 12, 7, 9, 8, 13]
#Cluster 2
# MS = [0, 17, 15, 13, 10, 9, 11, 21, 0, 7, 6, 3, 2, 5, 4, 8, 21, 0, 12, 16, 14, 18, 19, 1, 20, 21]
# Cluster 3
# MS = [0, 9, 21, 20, 15, 26, 28, 17, 29, 0, 27, 14, 6, 8, 3, 2, 7, 5, 1, 29, 0, 4, 19, 22, 18, 25, 24, 13, 29, 0, 23, 16, 11, 12, 10, 29]

# -----------------------------------------------------------------------------
# ------------------------ PARAMETERS OF THE MODEL ----------------------------
# -----------------------------------------------------------------------------

# Total number of Customers
n = instance.shape[0]-2

# Limiting capacity of each vehicle
Q = 35

# Service time at each customer node i
b = 5

# Setup cost of each vehicle
g = 40

# Expected Demand of each customer 
d = demand

# Time at which the service must finish
s = 480
    
# The pairwise cost of the customers
c = cost

# -----------------------------------------------------------------------------
# --------------------------- Genetic Algorithm -------------------------------
# -----------------------------------------------------------------------------

# Generate solutions
pop_size = 100 # one can increase this number based on the number of customers
population = np.zeros((pop_size, n), dtype=int)
pop_list = []
while True:
    pop = np.random.permutation(np.arange(n))
    if any((pop == x).all() for x in pop_list):
        continue
    pop_list.append(pop)
    
    if len(pop_list) == pop_size:
        break

population = np.asarray(pop_list)
population += 1   

# # Check if all the chromosomes are different in population
# new_p=[tuple(row) for row in population]
# K = np.unique(new_p,axis=0)
# K.shape

# Number of Generations
gen = 100

# Number of chromosomes selected for crossover
num_co = math.floor(pop_size/2)

# Number of chromosomes selected for mutation
num_mo = math.floor(pop_size/3)

for i in range(gen):
    print(i)
        
    # # Selection 1
    # # list of chromosomes to be selected for crossover
    # idx_co = random.sample(range(0, pop_size), num_co)
    # for j in range(1,num_co-1,2):
    #     child = Crossover(population[idx_co[j]], population[idx_co[j+1]])
    #     a = SolutionCost(child, Q, n, cost, g, MS)
    #     b = SolutionCost(population[idx_co[j]], Q, n, cost, g, MS)
    #     c = SolutionCost(population[idx_co[j+1]], Q, n, cost, g, MS)
                          
    #     if a < b:
    #         population[idx_co[j]]=child
    #     elif a < c:
    #         population[idx_co[j+1]]=child
    
    # # Selection 2
    # for j in range(1,pop_size-1,2):
    #     child = Crossover(population[j], population[j+1])
    #     a = SolutionCost(child, Q, n, cost, g, MS)
    #     b = SolutionCost(population[j], Q, n, cost, g, MS)
    #     c = SolutionCost(population[j+1], Q, n, cost, g, MS)
                          
    #     if a < b:
    #         population[j]=child
    #     elif a < c:
    #         population[j+1]=child
    
    # Selection 3 - Tournament Selection
    # pool_sz = math.floor(pop_size/2)
    # num_of_selections = num_co
    # for s in range(num_of_selections):
    #     sub_pop_1 = random.sample(range(pop_size), pool_sz)
    #     sub_pop_2 = random.sample(range(pop_size), pool_sz)
        
    #     best_sub_pop_1, cost_sub_pop_1, _ = BestSolution(population[sub_pop_1], Q, n, cost, g, MS)
    #     best_sub_pop_2, cost_sub_pop_2, _ = BestSolution(population[sub_pop_2], Q, n, cost, g, MS)
        
    #     best_idx_1 = sub_pop_1[best_sub_pop_1]
    #     best_idx_2 = sub_pop_2[best_sub_pop_2]
    #     child = Crossover(population[best_idx_1], population[best_idx_2])
    #     a = SolutionCost(child, Q, n, cost, g, MS)
                          
    #     if a < cost_sub_pop_1:
    #         population[best_idx_1]=child
    #     elif a < cost_sub_pop_2:
    #         population[best_idx_2]=child
            
    # # Selection 4 - Roulette Wheel selection
    costs_array = np.zeros((population.shape[0],))
    for i in range(population.shape[0]):
        costs_array[i] = SolutionCost(population[i], Q, n, cost, g, MS)
        
    costs_array = 1.0 / costs_array
    probabilities = costs_array / np.sum(costs_array)
    for i in range(num_co):
        parents = np.random.choice(range(pop_size), 2, p=probabilities)
        child = Crossover(population[parents[0]], population[parents[1]])
        a = SolutionCost(child, Q, n, cost, g, MS)
        b = SolutionCost(population[parents[0]], Q, n, cost, g, MS)
        c = SolutionCost(population[parents[1]], Q, n, cost, g, MS)
                          
        if a < b:
            population[parents[0]]=child
        elif a < c:
            population[parents[1]]=child
    
    # Mutation
    # list of chromosomes to be selected for mutation
    idx_co2 = random.sample(range(0, pop_size), num_mo)
    for j in range(num_mo):
        child = mutate(population[idx_co2[j]], 0.2)
        a = SolutionCost(child, Q, n, cost, g, MS)
        b = SolutionCost(population[idx_co2[j]], Q, n, cost, g, MS)
                      
        if a < b:
            population[idx_co2[j]]=child
     
    best_sol, best_cost, _ = BestSolution(population, Q, n, cost, g, MS)
    #print("Best solution id: ", best_sol)
    print("Best solution cost: ", best_cost)
    
    
s = list(population[best_sol])
new_s = [0] #This will be an array having the indices of depots where appropriate
total_demand = 0
for node in s:
    total_demand += d[node]
    if total_demand > Q:
        new_s.append(n+1)
        new_s.append(0)
        total_demand = 0
    new_s.append(node)
  
new_s.append(n+1)
route = []
counter = 1
for i in new_s:
    route.append(i)
    if i ==  n+1:
        print('Route: ', counter, ': ',route)
        route = []
        counter += 1 
        
bbb = SolutionCost(s, Q, n, cost, g, MS=MS)
print(bbb)
        
# best_bb = [2, 5, 4, 1, 3]
# best_bb = [3, 4, 6, 5, 7, 1, 8, 2]
# print(SolutionCost(best_bb, Q, n, cost, g, MS=MS))

end = time.time()

print( end - start)


