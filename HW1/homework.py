import os

INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"

# Preprocessing Cleanup
output_file_exists = os.path.isfile(OUTPUT_FILE)

if output_file_exists:
    os.remove(OUTPUT_FILE)
    print(f"Removed '{OUTPUT_FILE}' file")

with open(INPUT_FILE, "r") as input_file:
    SEARCH_ALGORITHM = input_file.readline().strip()
    UPHILL_ENERGY_LIMIT = int(input_file.readline())
    NUM_SAFE_LOCATIONS = int(input_file.readline())

    locations = {}
    for _ in range(NUM_SAFE_LOCATIONS):
        name, x, y, z = input_file.readline().split()

        name, x, y, z = name, int(x), int(y), int(z)
        locations[name] = {"coord": (x, y, z)}

    NUM_SAFE_PATH_SEGMENTS = input_file.readline().strip()

# Search Algorithms
def bfs_search():
    return "bfs"

def ucs_search():
    return "ucs"

def a_star_search():
    return "a star"


switcher = {
    "BFS": bfs_search,
    "UCS": ucs_search,
    "A*": a_star_search,
}

def switch(algorithm):
    return switcher.get(algorithm)()

result = switch(SEARCH_ALGORITHM)

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result)