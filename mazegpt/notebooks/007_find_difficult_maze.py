
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

EVAL_ROWS = 8000

with open("../data/mazes/oracle/data.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()[:EVAL_ROWS]]

# %%
from mazegpt.utils import Tile
from tqdm import tqdm

for i, row in tqdm(enumerate(dataset)):
    maze = row["maze"]
    directions = row["directions"]
    prompt = f"{maze};{directions}"
    maze_size = len(maze) + 1

    markers = row["markers"]

    if len(markers) == 0:
        continue
    
    marker_pos = [pos for _, pos, _ in markers]

    encoded = encode(prompt)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)
    # get logits
    logits, _ = model(encoded)
    # softmax
    logits = torch.softmax(logits, dim=-1).squeeze(0 )

    # mask to directions
    probabilities = logits[maze_size:]

    # Find the maximum probability for each position
    max_probabilities, _ = torch.max(probabilities, dim=1)
    
    threshold = 0.6
    # Create a mask for positions where the maximum probability is below the threshold
    low_confidence_mask = max_probabilities < threshold
    
    # Get the indices of low confidence positions
    low_confidence_positions = torch.nonzero(low_confidence_mask, as_tuple=True)[0]
    
    for pos in low_confidence_positions:
        relative_pos = pos.item()
        if relative_pos in marker_pos:
            print(i)
            print(prompt, low_confidence_positions.tolist())

            abs_pos = relative_pos + maze_size

            prompt = prompt[:abs_pos]

            display_maze(*parse_maze(prompt))



# %%
