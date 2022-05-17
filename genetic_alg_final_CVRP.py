# IMPLEMENTATION OF GENETIC ALGORITHM FOR THE CVRP PROBLEM

#----------------------- IMPORT REQUIRED PACKAGES -----------------------------

import numpy as np
import pandas as pd
import copy
import random
import math
import time


start = time.time()
# Number of Generations
gen = 200

# Generate solutions
pop_size = 300 # one can increase this number based on the number of customers

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
            print(pair, ' not in ', solution_pairs)
            
    print(add_cost)
    
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
instance = pd.Series([74]).append(pd.read_excel('Main Data VRSP.xlsx', 'IC3')['Node'])
instance = instance.append(pd.Series([74]))
demand = pd.Series([0]).append(pd.read_excel('Main Data VRSP.xlsx', 'IC3')['Demand_SM'])
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
# ------------------------ PARAMETERS OF THE MODEL ----------------------------
# -----------------------------------------------------------------------------

# Total number of Customers
n = instance.shape[0]-2

# Limiting capacity of each vehicle
Q = 35

# Service time at each customer node i
h = 5

# Setup cost of each vehicle
g = 40

# Expected Demand of each customer 
d = demand

# Time at which the service must finish
f0 = 480
    
# The pairwise cost of the customers
c = cost

# -----------------------------------------------------------------------------
# --------------------------- Genetic Algorithm -------------------------------
# -----------------------------------------------------------------------------

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
    #     a = SolutionCost(child, Q, n, cost, g)
    #     b = SolutionCost(population[idx_co[j]], Q, n, cost, g)
    #     c = SolutionCost(population[idx_co[j+1]], Q, n, cost, g)
                          
    #     if a < b:
    #         population[idx_co[j]]=child
    #     elif a < c:
    #         population[idx_co[j+1]]=child
    
    # # Selection 2
    # for j in range(1,pop_size-1,2):
    #     child = Crossover(population[j], population[j+1])
    #     a = SolutionCost(child, Q, n, cost, g)
    #     b = SolutionCost(population[j], Q, n, cost, g)
    #     c = SolutionCost(population[j+1], Q, n, cost, g)
                          
    #     if a < b:
    #         population[j]=child
    #     elif a < c:
    #         population[j+1]=child
    
    # # Selection 3 - Tournament Selection
    # pool_sz = math.floor(pop_size/2)
    # num_of_selections = num_co
    # for s in range(num_of_selections):
    #     sub_pop_1 = random.sample(range(pop_size), pool_sz)
    #     sub_pop_2 = random.sample(range(pop_size), pool_sz)
        
    #     best_sub_pop_1, cost_sub_pop_1, _ = BestSolution(population[sub_pop_1], Q, n, cost, g)
    #     best_sub_pop_2, cost_sub_pop_2, _ = BestSolution(population[sub_pop_2], Q, n, cost, g)
        
    #     best_idx_1 = sub_pop_1[best_sub_pop_1]
    #     best_idx_2 = sub_pop_2[best_sub_pop_2]
    #     child = Crossover(population[best_idx_1], population[best_idx_2])
    #     a = SolutionCost(child, Q, n, cost, g)
                          
    #     if a < cost_sub_pop_1:
    #         population[best_idx_1]=child
    #     elif a < cost_sub_pop_2:
    #         population[best_idx_2]=child
            
    # Selection 4 - Roulette Wheel selection
    costs_array = np.zeros((population.shape[0],))
    for i in range(population.shape[0]):
        costs_array[i] = SolutionCost(population[i], Q, n, cost, g)
        
    costs_array = 1.0 / costs_array
    probabilities = costs_array / np.sum(costs_array)
    for i in range(num_co):
        parents = np.random.choice(range(pop_size), 2, p=probabilities)
        child = Crossover(population[parents[0]], population[parents[1]])
        a = SolutionCost(child, Q, n, cost, g)
        b = SolutionCost(population[parents[0]], Q, n, cost, g)
        c = SolutionCost(population[parents[1]], Q, n, cost, g)
                          
        if a < b:
            population[parents[0]]=child
        elif a < c:            population[parents[1]]=child
    
    # Mutation
    # list of chromosomes to be selected for mutation
    idx_co2 = random.sample(range(0, pop_size), num_mo)
    for j in range(num_mo):
        child = mutate(population[idx_co2[j]], 0.2)
        a = SolutionCost(child, Q, n, cost, g)
        b = SolutionCost(population[idx_co2[j]], Q, n, cost, g)
                      
        if a < b:
            population[idx_co2[j]]=child
     
    
    best_sol, best_cost, _ = BestSolution(population, Q, n, cost, g)
    #print("Best solution id: ", best_sol)
    print("Best solution cost: ", best_cost) 
    
s = list(population[best_sol])
new_s = [0] #This will be an array havi ng the indices of depots where appropriate
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
        
bbb = SolutionCost(s, Q, n, cost, g)
print(bbb)

        
# best_bb = [1, 3, 2, 5, 4]
# best_bb = [3, 4, 6, 5, 1, 8, 2, 7]
# print(SolutionCost(best_bb, Q, n, cost, g))

end = time.time()

print( end - start)