import os
import heapq
from collections import deque

global SEARCH_ALGORITHM, UPHILL_ENERGY_LIMIT, start, goal, graph
INPUT_FILE = "/Users/mobolajiolawale/Documents/GitHub/CSCI_561/HW1/training-v2/input11.txt"
# INPUT_FILE = "sample_input.txt"
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

def heuristics(vertex_a, vertex_b):
    return get_distance(vertex_a, vertex_b, 3)

def build_search_graph(graph, relationships, algorithm):
    for relationship in relationships:
        vertex_a, vertex_b = relationship.strip().split()
        vertex_a_coord, vertex_b_coord = graph[vertex_a]['coord'], graph[vertex_b]['coord']

        if algorithm == "BFS":
            path_distance = 1
        elif algorithm == "UCS":
            path_distance = get_distance(vertex_a_coord, vertex_b_coord, 2)
        elif algorithm == "A*":
            path_distance = get_distance(vertex_a_coord, vertex_b_coord, 3)

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
    # visited_states = set()
    visited_states = {start: (0, 0)} # (path_distance, momentum)
    priority_cost_queue = [(0.0, start, [start], 0)]  # (path_distance, current_vertex_name, current_path_to_vertex, prev_energy)

    while priority_cost_queue:
        path_distance, current_vertex_name, path, prev_energy = heapq.heappop(priority_cost_queue)

        if current_vertex_name == goal:
            return ' '.join(path)

        momentum = abs(prev_energy) if prev_energy <= 0 else 0
        current_state = f"{current_vertex_name} {momentum}"

        # if current_state not in visited_states:
        #     visited_states.add(current_state)
        neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

        for neighbor_name, distance_to_neighbor in neighbors:
            current_location_coord = graph[current_vertex_name]['coord']
            next_location_coord = graph[neighbor_name]['coord']

            new_distance = visited_states[current_vertex_name][0] + distance_to_neighbor
            
            v_cost, v_momentum = visited_states.get(neighbor_name, (float('inf'), 0))
            if move_is_allowed(current_location_coord, next_location_coord, momentum)\
                and ( new_distance < v_cost or momentum == 0 and prev_energy < 0 ):
                
                visited_states[neighbor_name] = (new_distance, momentum)
                energy = get_req_energy(current_location_coord, next_location_coord)

                heapq.heappush(priority_cost_queue, (new_distance, neighbor_name, path + [neighbor_name], energy))

    return 'FAIL'

def ucs_search_v2(start, goal):
    visited_states = {} 
    priority_cost_queue = [(0.0, start, [start], 0)]  # (path_distance, current_vertex_name, current_path_to_vertex, prev_energy)

    while priority_cost_queue:
        path_distance, current_vertex_name, path, prev_energy = heapq.heappop(priority_cost_queue)

        if current_vertex_name == goal:
            return ' '.join(path)

        momentum = abs(prev_energy) if prev_energy <= 0 else 0
        current_state = (path_distance, momentum) # (path_distance, momentumValue)
        visited_states[current_vertex_name] = current_state

        neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

        for neighbor_name, distance_to_neighbor in neighbors:
            current_location_coord = graph[current_vertex_name]['coord']
            next_location_coord = graph[neighbor_name]['coord']

            # new_distance = visited_states[current_vertex_name][0] + distance_to_neighbor
            new_distance = path_distance + distance_to_neighbor
            prev_distance, prev_momentum = visited_states.get(neighbor_name, (float('inf'), 0))

            if move_is_allowed(current_location_coord, next_location_coord, momentum):
                # and ( new_distance < prev_distance or (current_vertex_name in visited_states and momentum > 0 and momentum > prev_momentum) ):
                
                visited_states[neighbor_name] = (new_distance, momentum)
                energy = get_req_energy(current_location_coord, next_location_coord)

                heapq.heappush(priority_cost_queue, (new_distance, neighbor_name, path + [neighbor_name], energy))

    return 'FAIL'

def a_star_search(start, goal):
    visited_states = set()
    visited = {start: (0, 0)} # (path_distance, momentum)
    priority_cost_queue = [(0.0, start, [start], 0)]  # (path_distance, current_vertex_name, current_path_to_vertex, prev_energy)

    while priority_cost_queue:
        path_distance, current_vertex_name, path, prev_energy = heapq.heappop(priority_cost_queue)

        if current_vertex_name == goal:
            return ' '.join(path)

        momentum = abs(prev_energy) if prev_energy <= 0 else 0
        current_state = f"{current_vertex_name} {momentum}"

        # if current_state not in visited_states:
        #     visited_states.add(current_state)
        neighbors = graph.get(current_vertex_name, {}).get('neighbors', [])

        for neighbor_name, distance_to_neighbor in neighbors:
            current_location_coord = graph[current_vertex_name]['coord']
    return 'FAIL'


switcher = {
    "BFS": bfs_search,
    "UCS": ucs_search,
    "A*": a_star_search,
}

def switch(algorithm, start, goal):
    return switcher.get(algorithm)(start, goal)

result = switch(SEARCH_ALGORITHM, 'start', 'goal')
print(len(result.split(' '))-1, '\n', result)

# with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
#     output_file.write(result)



