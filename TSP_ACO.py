import numpy as np
import pandas as pd
from collections import defaultdict
from haversine import haversine
import networkx as nx
import matplotlib.pyplot as plt
import random

# Define latitude and longitude for twenty major U.S. cities
cities = {
    'New York City': (40.72, -74.00),
    'Philadelphia': (39.95, -75.17),       
    'Baltimore': (39.28, -76.62),
    'Charlotte': (35.23, -80.85),
    'Memphis': (35.12, -89.97),
    'Jacksonville': (30.32, -81.70),
    'Houston': (29.77, -95.38),
    'Austin': (30.27, -97.77),
    'San Antonio': (29.53, -98.47),
    'Fort Worth': (32.75, -97.33),
    'Dallas': (32.78, -96.80),
    'San Diego': (32.78, -117.15),
    'Los Angeles': (34.05, -118.25),
    'San Jose': (37.30, -121.87),
    'San Francisco': (37.78, -122.42),    
    'Indianapolis': (39.78, -86.15),
    'Phoenix': (33.45, -112.07),       
    'Columbus': (39.98, -82.98), 
    'Chicago': (41.88, -87.63),
    'Detroit': (42.33, -83.05)
}

# Create a haversine distance matrix based on latitude-longitude coordinates
distance_matrix = defaultdict(dict)
for ka, va in cities.items():
    for kb, vb in cities.items():
        distance_matrix[ka][kb] = 0.0 if kb == ka else haversine((va[0], va[1]), (vb[0], vb[1])) 

# Convert distance diccionary into a dataframe        
distances = pd.DataFrame(distance_matrix)

city_names=list(distances.columns)    
distances=distances.values  
city_count = len(city_names)

# Create a graph
G=nx.Graph()

for ka, va in cities.items():
    for kb, vb in cities.items():
        G.add_weighted_edges_from({(ka,kb, distance_matrix[ka][kb])})

G.remove_edges_from(nx.selfloop_edges(G))
        
fig, ax = plt.subplots(figsize=(15,10))

# Reverse lat and long for correct visualization
reversed_dict = {key: value[::-1] for key, value in cities.items()}

# Create an independent shallow copy of the graph and attributes
H = G.copy()
 
# Draw the network
ax=nx.draw_networkx(
    H,
    pos=reversed_dict,
    with_labels=True,
    edge_color="gray",
    node_size=200,
    width=1,
)

plt.show()

def cost_function(path, distances):
    path_distance = 0
    for i in range(len(path) - 1):
        path_distance += distances[path[i], path[i+1]]
    return path_distance

def ant_tour(pheromones): 
    paths = np.empty((ants, city_count), dtype=int)
    for ant in range(ants):
        path = [0]
        unvisited_cities = list(range(1, city_count))

        while unvisited_cities:
            current_city = path[-1]
            probabilities = []

            for city in unvisited_cities:
                tau = pheromones[current_city, city] 
                eta = (1 / distances[current_city, city])
                probabilities.append((tau** alpha)*(eta ** beta))

            probabilities /= sum(probabilities)
            next_city = random.choices(unvisited_cities, weights=probabilities)[0]

            unvisited_cities.remove(next_city)
            path.append(next_city)

        paths[ant] = path
    
    return paths

def update_pheromones(distances, paths, pheromones):
    delta_pheromones = np.zeros((city_count, city_count))

    for i in range(ants):
        for j in range(city_count - 1):
            delta_pheromones[paths[i, j], paths[i, j+1]] += Q / distances[i]
        delta_pheromones[paths[i, -1], paths[i, 0]] += Q / distances[i]

    return (1 - evaporation_rate) * pheromones + delta_pheromones

def run_ACO(distances, ants, iterations, alpha, beta, evaporation_rate, Q):
    pheromones = np.ones((city_count, city_count))
    best_path = None
    best_distance = float('inf')

    for _ in range(iterations):
        # Generate paths for each ant
        paths =ant_tour(pheromones)

        distances_paths = np.array([cost_function(path, distances) for path in paths])
        min_idx = distances_paths.argmin()
        min_distance = distances_paths[min_idx]

        if min_distance < best_distance:
            best_distance = min_distance
            best_path = paths[min_idx]

        # Update pheromones
        update_pheromones(distances_paths, paths, pheromones)

    return best_path, best_distance

# ACO parameters
ants = 30
iterations = 100
alpha = 1
beta = 0.9
evaporation_rate = 0.5
Q = 100

# Run ACO with the predefined parameters
best_path, best_distance = run_ACO(distances, ants, iterations, alpha, beta, evaporation_rate, Q)
#print(best_path)
city_names_tour=[city_names[i] for i in best_path]
print(city_names_tour)
Route = " → ".join(city_names_tour)
print("Route:", Route)
print("Route length:", np.round(best_distance, 3))

fig, ax = plt.subplots(figsize=(15,10))

# Create an independent shallow copy of the graph and attributes
H = G.copy()

# Reverse lat and long for correct visualization
reversed_dict = {key: value[::-1] for key, value in cities.items()}

# Create edge list
edge_list =list(nx.utils.pairwise(city_names_tour))

# Draw closest edges on each node only
nx.draw_networkx_edges(H, pos=reversed_dict, edge_color="gray", width=0.5)
#print(reversed_dict)
#print(edge_list)
# Draw the route
ax=nx.draw_networkx(
    H,
    pos=reversed_dict,
    with_labels=True,
    edgelist=edge_list,
    edge_color="red",
    node_size=200,
    width=3,
)

plt.show()