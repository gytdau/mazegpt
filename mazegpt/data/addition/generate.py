# %%
from enum import Enum
import os
from typing import List, Tuple
import random
from gytis import imports
from mazegpt.utils import Markers, Tile, display_maze, display_maze_with_markers, serialize_sample



# generates sums of the form xxx+xxx=yyy, returns tuple of question and answer
def generate_sum(digits: int) -> Tuple[str, str]:
    x = random.randint(1, 10**digits - 1)
    y = random.randint(1, 10**digits - 1)
    question = f"{x:0{digits}d}+{y:0{digits}d}="
    answer = x + y
    return question, str(answer)

def insert_spaces(sum: Tuple[str, str]) -> str:
    question, answer = sum
    spaces = "   "
    return question + spaces + answer


# %%
import json
import tqdm
# Assume your functions are defined here

def main():
    quantity = 1_000_000

    file_path = os.path.join(os.path.dirname(__file__), "data.txt")
    # delete file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as file:
        for _ in tqdm.tqdm(range(quantity)):
            sum = generate_sum(3)
            sum = insert_spaces(sum)

            # Write the JSON string to a file, appending a newline to form the JSON Lines format
            file.write(sum + '\n')

main()

# %%
