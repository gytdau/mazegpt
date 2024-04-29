
# %% [markdown]
# Convenience file to sample the solution to a maze quickly.
from gytis import imports
from mazegpt.sample import model, encode, decode, device, stoi
import torch
from mazegpt.utils import display_maze, parse_maze
from mazegpt.notebooks.prepare_data import prepare_data
MARKER_TARGET = "mistake"

# %%
import json

EVAL_ROWS = 2000

with open("../data/mazes/eval/data.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()[:EVAL_ROWS]]

# %%
from mazegpt.utils import Tile
from tqdm import tqdm

def check_correct(grid: list[list[Tile]], path: str) -> bool:
    rows, cols = len(grid), len(grid[0])
    start_pos = None

    # Find the starting position
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == Tile.START:
                start_pos = (i, j)
                break
        if start_pos:
            break

    assert start_pos is not None, "No start position found"

    # Define the direction mappings
    directions = {
        'N': (-1, 0),
        'S': (1, 0),
        'W': (0, -1),
        'E': (0, 1)
    }

    # Traverse the path
    current_pos = start_pos
    for direction in path:
        if direction not in directions:
            return False

        row_offset, col_offset = directions[direction]
        new_row, new_col = current_pos[0] + row_offset, current_pos[1] + col_offset

        # Check if the new position is within the grid boundaries
        if not (0 <= new_row < rows and 0 <= new_col < cols):
            return False

        # Check if the new position is a wall
        if grid[new_row][new_col] == Tile.WALL:
            return False

        current_pos = (new_row, new_col)

    # Check if the final position is the end tile
    return grid[current_pos[0]][current_pos[1]] == Tile.END


correct = 0
total = 0
for i, row in tqdm(enumerate(dataset)):
    maze = row["maze"]
    prompt = f"{maze};"
    prompt_size = len(prompt)
    encoded = encode(prompt)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)
    completion_tokens = model.generate(encoded, max_new_tokens=120, temperature=0.1)
    completion = decode(completion_tokens.squeeze(0)[prompt_size:].tolist())
    # cut off at first ;
    seperator = ";"
    completion = completion[:completion.index(seperator)].strip()
    maze, directions = parse_maze(prompt + completion)
    # display_maze(maze, directions)
    correct += check_correct(maze, directions)
    total += 1

    if i % 100 == 0:
        print(f"Correct: {correct}/{total}")


print(f"Correct: {correct}/{total}")

# %%
