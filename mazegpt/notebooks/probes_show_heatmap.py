# %%
"""
Sample from a trained model
"""
from gytis import imports
from mazegpt.sample import model, encode, decode, itos, stoi, device
MARKER_TARGET = "fallible_goes_south"

# %%
import os
import torch
import torch.nn as nn
import json
from mazegpt.utils import display_maze, parse_maze
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

dataset_path = os.path.join(os.path.dirname(__file__), "../data/mazes/correctable/data.jsonl")

num_samples_for_probe = 10_000
with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f.readlines()[:num_samples_for_probe]]


inputs = []
targets = []
for i, row in enumerate(dataset):
    serialized = row["maze"] + ";" + "".join(row["directions"]) + ";\n"
    tokens = torch.tensor(encode(serialized))  # Ensure this is a tensor or convert it into one
    classes = torch.zeros(len(tokens), dtype=torch.long)

    marker_positions = [len(row["maze"]) + 1 + marker_pos for marker, marker_pos, _ in row["markers"] if marker == MARKER_TARGET]
    classes[marker_positions] = 1

    if len(marker_positions) == 0:
        continue

    inputs.append(tokens)
    targets.append(classes)


padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0).to(device)
padded_targets = pad_sequence(targets, batch_first=True, padding_value=0).to(device) 

test_proportion = 0.2


train_dataset = TensorDataset(padded_inputs[:int(len(padded_inputs) * (1 - test_proportion))], 
                              padded_targets[:int(len(padded_targets) * (1 - test_proportion))])
test_dataset = TensorDataset(padded_inputs[int(len(padded_inputs) * (1 - test_proportion)):], 
                             padded_targets[int(len(padded_targets) * (1 - test_proportion)):])

batch_size = 64  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"There are {len(padded_inputs)} samples in the dataset")

# %%
class GPTWithLinearProbe(nn.Module):
    def __init__(self, gpt_model, layer, num_classes):
        super().__init__()
        self.gpt = gpt_model
        self.layer = layer
        self.gpt.eval()  # Freeze the GPT model weights
        self.probe_layers = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, num_classes),
        )
    
    def forward(self, idx):
        _, _, hidden_states = self.gpt(idx, return_hidden_states=True)
        hidden_state = hidden_states[self.layer]  
        logits = self.probe_layers(hidden_state)
        return logits


class GPTWithProbe(nn.Module):
    def __init__(self, gpt_model, layer, num_classes):
        super().__init__()
        self.gpt = gpt_model
        self.layer = layer
        self.gpt.eval()  # Freeze the GPT model weights
        self.probe_layers = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, idx):
        _, _, hidden_states = self.gpt(idx, return_hidden_states=True)
        hidden_state = hidden_states[self.layer]  
        logits = self.probe_layers(hidden_state)
        return logits



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

example_row = dataset[132]

maze, directions = parse_maze(example_row["maze"] + ";" + "".join(example_row["directions"]) + ";\n")
markers = [(marker, pos, correct_move) for marker, pos, correct_move in example_row["markers"] if marker == MARKER_TARGET]


encoded_maze = encode(serialize_maze(maze))
mistake_abs_pos = 34
print(mistake_abs_pos)
print(f"Maze length: {len(encoded_maze)}")

token_id = mistake_abs_pos

def display_logits(logits_a):
    import matplotlib.pyplot as plt

    decoded_tokens = [repr(decoded) for _, decoded in itos.items()]
    softmax_values_a = [logits_a[token_idx].item() for token_idx, _ in itos.items()]

    plt.figure(figsize=(10, 6))
    width = 0.4  # the width of the bars
    r1 = np.arange(len(decoded_tokens))
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[decoded_tokens[int(i)] for i in r1], y=softmax_values_a,
                        name='Normal', marker_color='skyblue'))

    fig.update_layout(title_text='Next move probability',
                    xaxis=dict(title='Token'),
                    yaxis=dict(title='Probability', range=[0, 1]),
                    barmode='group')

    fig.show()

maze_size = len(maze)
mistake_prob = torch.zeros(maze_size, maze_size)
tokens = encode(serialize_sample(maze, directions))
tokens = tokens[:token_id]
tokens = torch.tensor(tokens).to(device)
display_maze(*parse_maze(decode(tokens.tolist())))

maze_without_end = copy.deepcopy(maze)
maze_without_end = [[x if x != Tile.END else Tile.EMPTY for x in row] for row in maze_without_end]

for x, y in tqdm(product(range(maze_size), range(maze_size))):
    new_maze = copy.deepcopy(maze_without_end)
    new_maze[x][y] = Tile.END
    tokens = encode(serialize_sample(new_maze, directions))
    tokens = tokens[:token_id]
    tokens = torch.tensor(tokens).to(device)
    # display_maze(*parse_maze(decode(tokens.tolist())))

    with torch.no_grad():  
        logits = probed_model(tokens.unsqueeze(0))
        logits = logits.squeeze(0)
        softmaxed = F.softmax(logits, dim=-1).cpu()

    mistake_prob[x, y] = softmaxed[-1][1].item()

import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
    z=mistake_prob,
    x=list(range(len(maze[0]))),
    y=list(range(len(maze))),
    colorscale='Viridis',
    zmin=0,  # Set minimum of z-axis scale
    zmax=1   # Set maximum of z-axis scale
))

fig.update_layout(
    title='Mistake Probability Heatmap',
    xaxis=dict(nticks=len(maze[0])),
    yaxis=dict(nticks=len(maze), autorange='reversed')  # Invert y-axis
)


fig.show()



# %%
