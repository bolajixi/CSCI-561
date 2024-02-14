import os
import time

start_time = time.time()

global ASSIGNED_COLOR, CURRENT_GAME_STATE
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"

CURRENT_GAME_STATE = []
X_GRID_SIZE, Y_GRID_SIZE = 12, 12
result = ""


# Preprocessing
# ---------------------------------------------------------------------------------------------------------------------------------------
output_file_exists = os.path.isfile(OUTPUT_FILE)

if output_file_exists:
    os.remove(OUTPUT_FILE)
    print(f"Removed '{OUTPUT_FILE}' file")

with open(INPUT_FILE, "r") as input_file:
    ASSIGNED_COLOR = input_file.readline().strip()
    GAME_PLAYTIME = input_file.readline().strip()
    STARTING_GAME_STATE = input_file.readlines()

    p1_time, p2_time = GAME_PLAYTIME.split()
    p1_time, p2_time = float(p1_time), float(p2_time)

    for row_string in STARTING_GAME_STATE:
        row = [cell for cell in row_string.strip()]
        CURRENT_GAME_STATE.append(row)

if X_GRID_SIZE != len(CURRENT_GAME_STATE[0]):
    raise Exception("X_GRID_SIZE does not match the number of columns in the game state")
elif Y_GRID_SIZE != len(CURRENT_GAME_STATE):
    raise Exception("Y_GRID_SIZE does not match the number of rows in the game state")


# Game Playing
# ---------------------------------------------------------------------------------------------------------------------------------------


elapsed_time = time.time() - start_time
print(f"\nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds \n\n{result}")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")