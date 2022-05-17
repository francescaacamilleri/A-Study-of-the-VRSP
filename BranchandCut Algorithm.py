from itertools import product
from mip import Model, xsum, BINARY, CONTINUOUS, minimize, OptimizationStatus
import numpy as np
import pandas as pd

# ---------------------- IMPORT DATA/INSTANCES --------------------------------

all_distances = pd.read_excel('Main Data VRSP.xlsx', 'distanceMatrix')
all_costs = pd.read_excel('Main Data VRSP.xlsx', 'CostMatrix', header=None)
instance = pd.Series([74]).append(pd.read_excel('Main Data VRSP.xlsx', 'I1')['Node'])
instance = instance.append(pd.Series([74]))
demand = pd.Series([0]).append(pd.read_excel('Main Data VRSP.xlsx', 'I1')['Demand_SM'])
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

# Total nuber of available vehicles
m = 10

# Index set of Customers
C = [i for i in range(1,n+1)]

# Index set of all the nodes
N = [0] + C + [n+1]

# Index set of vehicles
K = set(range(m)) 

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

# The cost from each node to any other node    
c = cost

# -----------------------------------------------------------------------------
# ----------------------------- CVRP MODEL ------------------------------------
# -----------------------------------------------------------------------------
 
md = Model('CVRP')


# Parameters needed for some constraints
# Big Ms
M = np.zeros((n+2,n+2)) # Big M numbers used in the Constraints (1)
for (i,j) in product(N,N):
    M[i,j] = s + Time[i][j] + b


# ------------------------ Desicion Variables ---------------------------------

x = [[[md.add_var(var_type=BINARY) for k in K] for j in N] for i in N]
u = [[md.add_var(var_type=CONTINUOUS) for k in K] for i in N]


# -------------------------- Objective ----------------------------------------

md.objective = minimize(xsum(g*x[0][i][k] for i in C for k in K) + xsum(c[i][j]*x[i][j][k] for i in N for j in N for k in K))


# ------------------------- Constraints ---------------------------------------

# A vehicle cannot go to the return depot and then visit a customer node
for (i,k) in product(N,K):
        md += x[n+1][i][k] == 0

# A vehicle cannot visit the same customer node twice
for (i,k) in product(N,K):
    md += x[i][i][k] == 0

# Only one vehicle must visit one customer node
for i in C:
    md += xsum(x[i][j][k] for k in K for j in N) == 1

# Each vehicle must start and end the route at the depot (if it is selected)
for k in K:
    md += xsum(x[0][j][k] for j in N) <= 1
    md += xsum(x[i][n+1][k] for i in N) <= 1
    
# Cohesive routes
for (i,k) in product(C,K):
    md += xsum(x[j][i][k] for j in N) - xsum(x[i][j][k] for j in N) == 0

# Subtour eliminations    
for (i,j,k) in product(N,N,K):
    md += u[j][k] >= u[i][k] + b + Time[i][j] - M[i][j] *(1 - x[i][j][k])

# Capacity Constraint
for k in K:
    md += xsum(d[i]*x[i][j][k] for i in C for j in N) <= Q
    
for (i,k) in product(N,K):
    md += u[i][k] >= 0


# md.optimize()

md.max_gap = 0.05
status = md.optimize(max_seconds=21600)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(md.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible:{}'.format(md.objective_value, md.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is:{}'.format(md.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('solution:')
    #for v in md.vars:
        #if abs(v.x) > 1e-6:# only printing non-zeros
        #print('{}:{}'.format(v.name, v.x))



selected_arcs = [(i,j,k+1) for (i,j,k) in product(N,N,K) if x[i][j][k].x >= 0.99]
print("selected arcs:{}".format(selected_arcs))

print("Objective value is: ",md.objective_value)

selected_arcs_no_vehicles = [(i,j) for (i,j,k) in product(N,N,K) if x[i][j][k].x >= 0.99]

route_veh = [[0] for i in range(m)]
for v in range(m):
    s = 0
    while(s < len(selected_arcs)):
        if selected_arcs[s][0] == route_veh[v][-1] and selected_arcs[s][2]==(v+1):
            route_veh[v].append(selected_arcs[s][1])
            s = 0
        s+=1
print("Routes:{}".format(route_veh))

arrival_veh = [[] for i in range(m)]   
idx = 0
for rv in route_veh:
    print('Arrival Time Vehicle: '+str(idx))
    for node in rv:
        arrival_veh[idx].append(u[node][idx].x)
        print('{:.2f}'.format(u[node][idx].x))
    idx+=1


# -----------------------------------------------------------------------------
# ----------------------------- VRSP MODEL ------------------------------------
# -----------------------------------------------------------------------------

# Actual Demand of each customer 
demand_new = pd.Series([0]).append(pd.read_excel('Main Data VRSP.xlsx', 'I1')['Demand_VRSP'])
demand_new = np.asarray(demand_new.append(pd.Series([0])))
d_new = demand_new

# Cost of deviating from the Master Schedule
F = 4

# ------------------------------ RESCHEDULING MODEL ---------------------------
md = Model('VRSP')

# Parameters needed for some constraints (Big Ms)
M = np.zeros((n+2,n+2)) # Big M numbers used in the Constraints (1)
for (i,j) in product(N,N):
    M[i,j] = s + Time[i][j] + b



# ------------------------ Desicion Variables ---------------------------------

x = [[[md.add_var(var_type=BINARY) for k in K] for j in N] for i in N]
u = [[md.add_var(var_type=CONTINUOUS) for k in K] for i in N]


# -------------------------- Objective ----------------------------------------


md.objective = minimize(xsum(g*x[0][i][k] for i in C for k in K) + xsum(c[i][j]*x[i][j][k] for i in N for j in N for k in K) + F*xsum(1-xsum(x[i][j][k] for k in K) for (i,j) in selected_arcs_no_vehicles))


# ------------------------- Constraints ---------------------------------------

# A vehicle cannot go to the return depot and then visit a customer node
for (i,k) in product(N,K):
        md += x[n+1][i][k] == 0

# A vehicle cannot visit the same customer node twice
for (i,k) in product(N,K):
    md += x[i][i][k] == 0

# Only one vehicle must visit one customer node
for i in C:
    md += xsum(x[i][j][k] for k in K for j in N) == 1
    
# Each vehicle must start and end the route at the depot (if it is selected)
for k in K:
    md += xsum(x[0][j][k] for j in N) <= 1
    md += xsum(x[i][n+1][k] for i in N) <= 1
    
# Cohesive routes
for (i,k) in product(C,K):
    md += xsum(x[j][i][k] for j in N) - xsum(x[i][j][k] for j in N) == 0

# Subtour eliminations    
for (i,j,k) in product(N,N,K):
    md += u[j][k] >= u[i][k] + b + Time[i][j] - M[i][j] *(1 - x[i][j][k])

# Capacity Constraint
for k in K:
    md += xsum(d_new[i]*x[i][j][k] for i in C for j in N) <= Q
    
for (i,k) in product(N,K):
    md += u[i][k] >= 0


# md.optimize()

md.max_gap = 0.05
status = md.optimize(max_seconds=21600)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(md.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible:{}'.format(md.objective_value, md.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is:{}'.format(md.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('solution:')
    #for v in md.vars:
        #if abs(v.x) > 1e-6:# only printing non-zeros
        #print('{}:{}'.format(v.name, v.x))



selected_arcs_2 = [(i,j,k+1) for (i,j,k) in product(N,N,K) if x[i][j][k].x >= 0.99]
print("selected arcs_2:{}".format(selected_arcs_2))

print("Objective value is: ",md.objective_value)

route_veh = [[0] for i in range(m)]
for v in range(m):
    s = 0
    while(s < len(selected_arcs_2)):
        if selected_arcs_2[s][0] == route_veh[v][-1] and selected_arcs_2[s][2]==(v+1):
            route_veh[v].append(selected_arcs_2[s][1])
            s = 0
        s+=1
print("Routes:{}".format(route_veh))

arrival_veh = [[] for i in range(m)]   
idx = 0
for rv in route_veh:
    print('Arrival Time Vehicle: '+str(idx))
    for node in rv:
        arrival_veh[idx].append(u[node][idx].x)
        print('{:.2f}'.format(u[node][idx].x))
    idx+=1
