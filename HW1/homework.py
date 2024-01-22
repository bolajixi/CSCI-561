import os
from collections import deque

global SEARCH_ALGORITHM, start, goal, locations_graph
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"

# Preprocessing Cleanup
# ---------------------------------------------------------------------------------------------------------------------------------------
output_file_exists = os.path.isfile(OUTPUT_FILE)

if output_file_exists:
    os.remove(OUTPUT_FILE)
    print(f"Removed '{OUTPUT_FILE}' file")

with open(INPUT_FILE, "r") as input_file:
    SEARCH_ALGORITHM = input_file.readline().strip()
    UPHILL_ENERGY_LIMIT = int(input_file.readline())
    NUM_SAFE_LOCATIONS = int(input_file.readline())

    locations_graph = {}
    for _ in range(NUM_SAFE_LOCATIONS):
        name, x, y, z = input_file.readline().split()

        name, x, y, z = name, int(x), int(y), int(z)
        locations_graph[name] = {'coord': (x, y, z), 'neighbors': []}

    NUM_SAFE_PATH_SEGMENTS = input_file.readline().strip()
    relationships = input_file.readlines()

    start = locations_graph['start']
    goal = locations_graph['goal']

# ---------------------------------------------------------------------------------------------------------------------------------------
def euclidean_distance_2d(point_a, point_b):
    x1, y1, z1 = point_a
    x2, y2, z2 = point_b
    return ( (x1 - x2)**2 + (y1 - y2)**2 )**0.5

def euclidean_distance_3d(point_a, point_b):
    x1, y1, z1 = point_a
    x2, y2, z2 = point_b
    return ( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )**0.5

# ---------------------------------------------------------------------------------------------------------------------------------------
def build_search_tree(locations_graph, relationships, algorithm):
    for relationship in relationships:
        start_vertex, end_vertex = relationship.strip().split()

        if algorithm == "BFS":
            weight = 1
        elif algorithm == "UCS":
            weight = euclidean_distance_2d(locations_graph[start_vertex]['coord'], locations_graph[end_vertex]['coord'])
        elif algorithm == "A*":
            weight = euclidean_distance_3d(locations_graph[start_vertex]['coord'], locations_graph[end_vertex]['coord'])

        locations_graph[start_vertex]['neighbors'].append(end_vertex)
        locations_graph[end_vertex]['neighbors'].append(start_vertex)

build_search_tree(locations_graph, relationships, SEARCH_ALGORITHM)

# Search Algorithms
# ---------------------------------------------------------------------------------------------------------------------------------------
def bfs_search(start, goal):
    visited = set()
    queue = deque( [(start, [start])] )

    while queue:
        current_vertex_name, path = queue.popleft()

        if current_vertex_name == goal:
            return ' '.join(path)

        if current_vertex_name not in visited:
            visited.add(current_vertex_name)

            neighbors = locations_graph.get(current_vertex_name, {}).get('neighbors', [])

            for neighbor_name in neighbors:
                if neighbor_name not in visited: # Is this check really necessary?
                    queue.append((neighbor_name, path + [neighbor_name]))

    return 'FAIL'

def ucs_search(start, goal):

    return 'FAIL'

def a_star_search(start, goal):

    return 'FAIL'


switcher = {
    "BFS": bfs_search,
    "UCS": ucs_search,
    "A*": a_star_search,
}

def switch(algorithm, start, goal):
    return switcher.get(algorithm)(start, goal)

result = switch(SEARCH_ALGORITHM, 'start', 'goal')
print(result)

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result)