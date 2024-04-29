# %%
"""
Sample from a trained model
"""
from gytis import imports
from mazegpt.notebooks.prepare_data import prepare_data
from mazegpt.sample import model, encode, decode, itos, stoi, device

# %% [markdown]
# ## Linear Probes
# We're training a probe to predict whether the current move is a marker_predicted.
# %%
MARKER_TARGET = "mistake"
dataset, train_loader, test_loader, get_ground_truth = prepare_data("../data/mazes/correctable/data.jsonl", 100_000, MARKER_TARGET)

# %%
from torch.optim import AdamW
from mazegpt.notebooks.probes import GPTWithLinearProbe
import torch
from torch.nn import functional as F

ProbeModel = GPTWithLinearProbe


from tqdm import tqdm

def train_model_at_layer(layer):
    print("---")
    print(f"Training probe at layer {layer}")
    probed_model = ProbeModel(model, layer=layer, num_classes=2).to(device)
    optimizer = AdamW(probed_model.probe_layers.parameters(), lr=0.01)
    EPOCHS = 10
    # Training loop
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
    
    return probed_model

probes = []
for layer in [5]:
    probes.append([layer, train_model_at_layer(layer)])
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mazegpt.utils import display_maze, display_maze_with_markers, parse_maze, serialize_sample

from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter
import plotly.graph_objects as go

# Ensure the model is in evaluation mode


def test_for_target(probed_model, target, layer_id):
    for test_id in range(100):
        example, example_target = test_dataset[test_id]
        example_target = example_target.cpu()
        example_row = dataset[len(train_dataset) + test_id]

        # Get the model's predictions for this example
        with torch.no_grad():  # No need to track gradients here
            example_logits = probed_model(example.unsqueeze(0)).squeeze(0)
            softmaxed = F.softmax(example_logits, dim=-1).cpu()

        token_positions = list(range(example.size(0)))

        maze, directions = parse_maze(example_row["maze"] + ";" + "".join(example_row["directions"]) + ";\n")
        markers = [(marker, pos) for marker, pos in example_row["markers"] if marker == target]

        if len(markers) == 0:
            continue

        seperator = encode(";")[0]
        maze_end, directions_end = [v.item() for v in (example == seperator).nonzero()]

        marker_predicted = softmaxed[:, 1]
        not_marker_predicted = softmaxed[:, 0]
        result_signal = list(marker_predicted[maze_end+1:directions_end].cpu().tolist())

        # Generate x values representing each token position


        # Create the plot
        fig = make_subplots(rows=2, cols=1, subplot_titles=(f'Probe prediction', 'Ground Truth'), 
                            vertical_spacing=0.1, shared_xaxes=True)

        fig.add_trace(go.Scatter(x=token_positions, y=marker_predicted, mode='lines+markers',
                                marker=dict(size=5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=token_positions, y=example_target, mode='lines+markers',
                                marker=dict(size=5)), row=2, col=1)

        fig.update_layout(title=f'L{layer_id} - `{target}` on Example {test_id}',
                            xaxis_title='Token Position',
                            yaxis_title='Probability', showlegend=False)
        fig.update_xaxes(side="bottom")
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.show()

        display_maze(maze, directions, signal=result_signal)


for (layer, probe) in probes:
    probe.eval()
    test_for_target(probe, MARKER_TARGET, layer)
# %%
from mazegpt.utils import display_maze, display_maze_with_markers, parse_maze, serialize_sample

test_id = 79
probe_id = 0
token_id = 37

row = dataset[test_id]

maze, directions = parse_maze(row["maze"] + ";" + "".join(row["directions"]) + ";\n")
markers = [(marker, pos, correct_move) for marker, pos, correct_move in row["markers"] if marker == MARKER_TARGET]

tokens = encode(serialize_sample(maze, directions))
# only first tokens
example_trimmed = tokens[:token_id]

maze, directions = parse_maze(decode(example_trimmed))
display_maze(maze, directions)
def display_logits(logits_a, logits_b,):
    import matplotlib.pyplot as plt

    decoded_tokens = [repr(decoded) for _, decoded in itos.items()]
    softmax_values_a = [logits_a[token_idx].item() for token_idx, _ in itos.items()]
    softmax_values_b = [logits_b[token_idx].item() for token_idx, _ in itos.items()]

    plt.figure(figsize=(10, 6))
    width = 0.4  # the width of the bars
    r1 = np.arange(len(decoded_tokens))
    r2 = [x + width for x in r1]
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[decoded_tokens[int(i)] for i in r1], y=softmax_values_a,
                         name='Normal', marker_color='skyblue'))
    fig.add_trace(go.Bar(x=[decoded_tokens[int(i)] for i in r2], y=softmax_values_b,
                         name='Intervention', marker_color='orange'))

    fig.update_layout(title_text='Next move probability',
                      xaxis=dict(title='Token'),
                      yaxis=dict(title='Probability', range=[0, 1]),
                      barmode='group')

    fig.show()


# Add the probe's vector to the residual stream
# new tensor
steering_vector = torch.zeros(model.config.n_layer, model.config.n_embd).to(device)
steering_vector[5] = probe.probe_layers[0].weight[0] * 400


with torch.no_grad():  
    example_logits, _ = model(example_trimmed.unsqueeze(0))
    example_logits = example_logits.squeeze(0).squeeze(0)
    softmaxed = F.softmax(example_logits, dim=-1).cpu()


with torch.no_grad():  
    steered_logits, _ = model(example_trimmed.unsqueeze(0), add_activations=steering_vector)
    steered_logits = steered_logits.squeeze(0).squeeze(0)
    steered_softmaxed = F.softmax(steered_logits, dim=-1).cpu()

display_logits(softmaxed, steered_softmaxed)
# %%
