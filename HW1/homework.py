import os
import heapq
import time
from collections import deque

start_time = time.time()

global SEARCH_ALGORITHM, UPHILL_ENERGY_LIMIT, start, goal, graph
# INPUT_FILE = "/Users/mobolajiolawale/Documents/GitHub/CSCI_561/HW1/training-v2/input39.txt"
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

    graph = {}
    for _ in range(NUM_SAFE_LOCATIONS):
        name, x, y, z = input_file.readline().split()

        name, x, y, z = name, int(x), int(y), int(z)
        graph[name] = {'coord': (x, y, z), 'neighbors': []}

    NUM_SAFE_PATH_SEGMENTS = input_file.readline().strip()
    relationships = input_file.readlines()

    start = graph['start']
    goal = graph['goal']

# Helper Functions
# ---------------------------------------------------------------------------------------------------------------------------------------
def get_req_energy(vertex_a, vertex_b):
    x1, y1, z1 = vertex_a
    x2, y2, z2 = vertex_b

    return z2 - z1

def get_distance(vertex_a, vertex_b, dimension):
    x1, y1, z1 = vertex_a
    x2, y2, z2 = vertex_b
    
    if dimension == 2:
        return ( (x1 - x2)**2 + (y1 - y2)**2 )**0.5
    elif dimension == 3:
        return ( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )**0.5

def move_is_allowed(current_vertex, next_vertex, momentum=0):
    required_energy = get_req_energy(current_vertex, next_vertex)

    current_move_is_downhill = True if required_energy < 0 else False
    if current_move_is_downhill:
        return True

    if UPHILL_ENERGY_LIMIT >= required_energy or UPHILL_ENERGY_LIMIT + momentum >= required_energy:
        return True

    return False

def heuristics(type, vertex, goal):
    if type == "euclidean":
        return get_distance(vertex, goal, 2)

def build_search_graph(graph, relationships, algorithm):
    for relationship in relationships:
        vertex_a, vertex_b = relationship.strip().split()
        vertex_a_coord, vertex_b_coord = graph[vertex_a]['coord'], graph[vertex_b]['coord']

        path_distance = "_"
            
        if algorithm == "UCS":
            path_distance = get_distance(vertex_a_coord, vertex_b_coord, 2)
        elif algorithm == "A*":
            path_distance = get_distance(vertex_a_coord, vertex_b_coord, 3)

        if vertex_a != vertex_b:
            graph[vertex_a]['neighbors'].append([vertex_b, path_distance])
            graph[vertex_b]['neighbors'].append([vertex_a, path_distance])

build_search_graph(graph, relationships, SEARCH_ALGORITHM)

# Search Algorithms
# ---------------------------------------------------------------------------------------------------------------------------------------
def bfs_search(start, goal):
    visited_states = set()

    queue = deque( [(start, [start], 0)] ) # (current_vertex_name, current_path_to_vertex, prev_energy)

    while queue:
        current_vertex_name, path, prev_energy = queue.popleft()

        if current_vertex_name == goal:
            return ' '.join(path)

        momentum = abs(prev_energy) if prev_energy <= 0 else 0
        current_state = f"{current_vertex_name} {momentum}"

        if current_state not in visited_states:
            visited_states.add(current_state)
            neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

            for neighbor_name, _ in neighbors:
                current_vertex_coord = graph[current_vertex_name]['coord']
                next_vertex_coord = graph[neighbor_name]['coord']

                if move_is_allowed(current_vertex_coord, next_vertex_coord, momentum):
                    prev_energy = get_req_energy(current_vertex_coord, next_vertex_coord)
                    queue.append((neighbor_name, path + [neighbor_name], prev_energy))

    return 'FAIL'

def ucs_search(start, goal):
    visited_states = {('start', 0): 0}
    priority_cost_queue = [(0.0, start, [start], 0)]  # (path_distance, current_vertex_name, current_path_to_vertex, prev_energy)

    while priority_cost_queue:
        path_distance, current_vertex_name, path, current_momentum = heapq.heappop(priority_cost_queue)

        if current_vertex_name == goal:
            return ' '.join(path)

        neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

        for neighbor_name, distance_to_neighbor in neighbors:
            current_location_coord = graph[current_vertex_name]['coord']
            next_location_coord = graph[neighbor_name]['coord']

            if move_is_allowed(current_location_coord, next_location_coord, current_momentum):
                new_distance = path_distance + distance_to_neighbor

                energy = get_req_energy(current_location_coord, next_location_coord)
                next_momentum = abs(energy) if energy <= 0 else 0
                
                current_state = (neighbor_name, next_momentum)
                if current_state in visited_states and new_distance >= visited_states[current_state]:
                    continue

                visited_states[current_state] = new_distance
                heapq.heappush(priority_cost_queue, (new_distance, neighbor_name, path + [neighbor_name], next_momentum))

    return 'FAIL'

def a_star_search(start, goal):
    start_coord = graph[start]['coord']
    goal_coord = graph[goal]['coord']

    start_heuristic_value = heuristics('euclidean', start_coord, goal_coord)
    visited_states = {('start', 0): 0}
    priority_cost_queue = [(start_heuristic_value, 0.0, start, [start], 0)] #(f_cost, path_distance, current_vertex_name, current_path_to_vertex, prev_energy)

    while priority_cost_queue:
        _, path_distance, current_vertex_name, path, current_momentum = heapq.heappop(priority_cost_queue)

        if current_vertex_name == goal:
            return ' '.join(path)

        neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

        for neighbor_name, distance_to_neighbor in neighbors:
            current_location_coord = graph[current_vertex_name]['coord']
            next_location_coord = graph[neighbor_name]['coord']

            if move_is_allowed(current_location_coord, next_location_coord, current_momentum):
                heuristic_value = heuristics('euclidean', next_location_coord, goal_coord)
                new_distance = path_distance + distance_to_neighbor
                f_cost = new_distance + heuristic_value

                energy = get_req_energy(current_location_coord, next_location_coord)
                next_momentum = abs(energy) if energy <= 0 else 0
                
                current_state = (neighbor_name, next_momentum)
                if (current_state in visited_states) and (new_distance >= visited_states[current_state]):
                    continue

                visited_states[current_state] = new_distance
                heapq.heappush(priority_cost_queue, (f_cost, new_distance, neighbor_name, path + [neighbor_name], next_momentum))

    return 'FAIL'

switcher = {
    "BFS": bfs_search,
    "UCS": ucs_search,
    "A*": a_star_search,
}

def switch(algorithm, start, goal):
    return switcher.get(algorithm)(start, goal)

result = switch(SEARCH_ALGORITHM, 'start', 'goal')

elapsed_time = time.time() - start_time
print(f"Path Length = {len(result.split(' ')) - 1} \nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds \n\n{result}")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")