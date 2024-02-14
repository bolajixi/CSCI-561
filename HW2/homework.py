import os
import heapq
import time
from collections import deque

start_time = time.time()

INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"


elapsed_time = time.time() - start_time
print(f"\nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")