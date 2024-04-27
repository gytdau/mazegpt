# %%
"""
Sample from a trained model
"""
from gytis import imports
from mazegpt.notebooks.prepare_data import prepare_data
from mazegpt.sample import model, encode, decode, itos, stoi, device
MARKER_TARGET = "mistake"

# %%
dataset, train_loader, test_loader, get_ground_truth = prepare_data("../data/mazes/correctable/data.jsonl", 100_000, MARKER_TARGET)

# %%
from torch.optim import AdamW
import torch
from mazegpt.notebooks.probes import GPTWithLinearProbe, GPTWithProbe
from torch.nn import functional as F

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

for test_id in range(10):#[93]:
    example_row = dataset[test_id]

    maze, directions = parse_maze(example_row["maze"] + ";" + "".join(example_row["directions"]) + ";\n")
    markers = [(marker, pos, correct_move) for marker, pos, correct_move in example_row["markers"] if marker == MARKER_TARGET]


    encoded_maze = encode(serialize_maze(maze))
    mistake_abs_pos = 111
    print(test_id)
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


    for x, y in product(range(maze_size), range(maze_size)):
        maze_without_end = copy.deepcopy(maze)
        maze_without_end = [[x if x != Tile.END else Tile.EMPTY for x in row] for row in maze_without_end]
        new_maze = copy.deepcopy(maze_without_end)
        new_maze[x][y] = Tile.END
        # new_maze[x][y] = Tile.WALL
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
        zmax=1,  # Set maximum of z-axis scale
        showscale=True,
        hoverongaps=False,
    ))

    fig.update_layout(
        title='Mistake probe after moving the end point',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(tickmode='linear', tick0=1, dtick=1, autorange='reversed'),  # Invert y-axis and set ticks on every tile
        height=500,
        width=480,
    )


    fig.show()



    # %%
