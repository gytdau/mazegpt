# %%
"""
Sample from a trained model
"""
from gytis import imports
from mazegpt.notebooks.prepare_data import prepare_data
from mazegpt.sample import model, encode, decode, itos, stoi, device
MARKER_TARGET = "fallible_goes_south"
# %%

dataset, train_loader, test_loader, get_ground_truth = prepare_data("../data/mazes/correctable/data.jsonl", 10_000, MARKER_TARGET)

# %%
from torch.optim import AdamW
from mazegpt.notebooks.probes import GPTWithLinearProbe
import torch.nn.functional as F

ProbeModel = GPTWithLinearProbe

from tqdm import tqdm

probed_model = ProbeModel(model, layer=5, num_classes=2).to(device)

def train(probed_model):
    optimizer = AdamW(probed_model.probe_layers.parameters(), lr=0.01)
    EPOCHS = 20

    def train_linear_probe():
        for epoch in range(EPOCHS):
            probed_model.probe_layers.train()
            total_loss = 0
            for inputs, targets in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = probed_model(inputs)
                outputs = outputs.transpose(1, 2)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Estimate on test set
            probed_model.probe_layers.eval()
            test_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = probed_model(inputs)
                    outputs = outputs.transpose(1, 2)
                    loss = F.cross_entropy(outputs, targets)
                    test_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}")
    
    train_linear_probe()
    
train(probed_model)
# %%

import numpy as np
from mazegpt.utils import display_maze, parse_maze, serialize_maze, serialize_sample, Tile
import copy
from tqdm import tqdm
from itertools import product
import torch

for row_id in range(10):
    row = dataset[row_id]

    maze, directions = parse_maze(row["maze"] + ";" + "".join(row["directions"]) + ";\n")
    markers = [(marker, pos, correct_move) for marker, pos, correct_move in row["markers"] if marker == MARKER_TARGET]


    encoded_maze = encode(serialize_maze(maze))
    maze_tokenized_length = len(encoded_maze)
    directions_length = len(directions)


    maze_size = len(maze)
    mistake_prob = torch.zeros(maze_size, maze_size)
    tokens = encode(serialize_sample(maze, directions))
    tokens = torch.tensor(tokens).to(device)

    # Ground truth
    print(f"Ground truth for {row_id}")
    classes = get_ground_truth(tokens, row)
    display_maze(maze, directions, classes.tolist()[maze_tokenized_length:])


    # Prediction
    signal = []
    for token_id in range(directions_length):
        tokens_until_token_id = encode(serialize_sample(maze, directions))
        
        cutoff = maze_tokenized_length + token_id
        tokens_until_token_id = tokens[:cutoff]
        tokens_until_token_id = torch.tensor(tokens).to(device)

        with torch.no_grad():  
            logits = probed_model(tokens_until_token_id.unsqueeze(0))
            logits = logits.squeeze(0)
            softmaxed = F.softmax(logits, dim=-1).cpu()

        signal.append(softmaxed[-1][1].item())


    print(f"Prediction for {row_id}")
    display_maze(maze, directions, signal)


# %%
