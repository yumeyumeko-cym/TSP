from optalgotools.problems import TSP
from optalgotools.algorithms import SimulatedAnnealing
import matplotlib.pyplot as plt
# data can be found on https://github.com/coin-or/jorlib/tree/master/jorlib-core/src/test/resources/tspLib/tsp
# the following link is the permalink for berlin52
berlin52_tsp_url = 'https://raw.githubusercontent.com/coin-or/jorlib/b3a41ce773e9b3b5b73c149d4c06097ea1511680/jorlib-core/src/test/resources/tspLib/tsp/berlin52.tsp'
# Create a tsp object for Belin52
# for a complete list of the suppored params, check the TSP class in the tsp.py file
berlin52_tsp = TSP(load_tsp_url=berlin52_tsp_url, gen_method='mutate', rand_len=True, init_method='random')
# Create a simulated annealing model
sa = SimulatedAnnealing(max_iter=1200, max_iter_per_temp=500, initial_temp=150, final_temp=0.01, cooling_schedule='linear', debug=1)
# Get an initial random solution and check its length 
# sa.init_annealing(berlin52_tsp,'random')
# sa.val_cur
# Run SA and eval the best solution distance
sa.run(berlin52_tsp, repetition=1)
print(sa.val_allbest)
# plot the solution
berlin52_tsp.plot(sa.s_best)