import numpy as np
from minisom import MiniSom
from collections import defaultdict
from haversine import haversine
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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

#print(distance_matrix)
# Convert distance diccionary into a dataframe        
distances = pd.DataFrame(distance_matrix)
#print(distances)

city_names = list(distances.columns)    
distances = distances.values  
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

# Function to create a distance matrix (optional for visualization)
def create_distance_matrix(cities):
    distance_matrix = defaultdict(dict)
    for ka, va in cities.items():
        for kb, vb in cities.items():
            distance_matrix[ka][kb] = haversine(va, vb)
    return pd.DataFrame(distance_matrix)

# Extract city coordinates into a NumPy array

coords = np.array(list(cities.values()))

# Initialize MiniSom
som_size = 20  # Adjust based on requirements
som = MiniSom(som_size, 1, 2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(coords)

# Train the SOM
iterations = 100000
som.train_random(coords, iterations)

# Extract the winning neurons' coordinates for each city
win_map = som.win_map(coords)

# Construct the route
route = []
for position in sorted(win_map.keys()):
    route.extend(win_map[position])

# Convert the route back to city names
route_cities = [list(cities.keys())[list(cities.values()).index(tuple(r))] for r in route]

# Calculate total route length
route_length = 0
for i in range(len(route_cities) - 1):
    city_a = route_cities[i]
    city_b = route_cities[i + 1]
    route_length += haversine(cities[city_a], cities[city_b])

# Print the route and its length
print("Route:", " -> ".join(route_cities))
print("Route Length: {:.2f} km".format(route_length))

# Optional: Visualization
# Assuming create_distance_matrix() is defined
distance_matrix = create_distance_matrix(cities)
plt.imshow(distance_matrix, interpolation='nearest')
plt.colorbar()
plt.show()

fig, ax = plt.subplots(figsize=(15,10))

# Create an independent shallow copy of the graph and attributes
H = G.copy()

# Reverse lat and long for correct visualization
reversed_dict = {key: value[::-1] for key, value in cities.items()}

# Create edge list
edge_list =list(nx.utils.pairwise(route_cities))

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