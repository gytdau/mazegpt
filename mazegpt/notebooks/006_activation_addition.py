# %%
from gytis import imports
from mazegpt.sample import model, encode, decode, itos, stoi, device, block_size

# %% [markdown]
# ## Activation Additions
import os
import torch
import torch.nn as nn
import json
from mazegpt.utils import display_maze, parse_maze
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

dataset_path = os.path.join(os.path.dirname(__file__), "../data/mazes/correctable/data.jsonl")

num_samples_for_probe = 500_000
with open(dataset_path, "r") as f:
    raw_dataset = [json.loads(line) for line in f.readlines()[:num_samples_for_probe]]

# discard all mazes that have no mistakes
dataset = [row for row in raw_dataset if len([marker for marker, _, _ in row["markers"] if marker == MARKER_TARGET]) > 0]

MARKER_TARGET = "mistake"
inputs = []
targets = []
for i in tqdm(range(len(dataset))):
    row = dataset[i]

    maze = row["maze"]
    directions = row["directions"]
    markers = row["markers"]
    serialized = maze + ";" + directions + ";\n"
    tokens = torch.tensor(encode(serialized)) 
    tokens = torch.cat([tokens, torch.zeros(block_size - len(tokens), dtype=torch.long)])

    target_markers = [(marker, marker_pos, correct_token) for marker, marker_pos, correct_token in markers if marker == MARKER_TARGET]

    if len(target_markers) == 0:
        continue

    target = tokens.clone()
    target = torch.roll(target, 1)

    for marker, marker_pos, correct_token in target_markers:
        abs_pos = len(maze) + 1 + marker_pos
        target[abs_pos] = encode(correct_token)[0]

    inputs.append(tokens)
    targets.append(target)


padded_inputs = torch.stack(inputs).to(device)
padded_targets = torch.stack(targets).to(device)

test_proportion = 0.2

train_dataset = TensorDataset(padded_inputs[:int(len(padded_inputs) * (1 - test_proportion))], 
                              padded_targets[:int(len(padded_targets) * (1 - test_proportion))])
test_dataset = TensorDataset(padded_inputs[int(len(padded_inputs) * (1 - test_proportion)):], 
                             padded_targets[int(len(padded_targets) * (1 - test_proportion)):])

batch_size = 64  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%

class GPTWithActivationAddition(nn.Module):
    def __init__(self, gpt_model, layer):
        super().__init__()
        self.gpt = gpt_model
        self.layer = layer
        self.gpt.eval()  # Freeze the GPT model weights
        self.activation_layers = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, 1),
        )
    
    def forward(self, idx):
        steering_vector = torch.zeros(model.config.n_layer, model.config.n_embd).to(device)
        steering_vector[self.layer] = self.activation_layers[0].weight[0]
        logits, _ = self.gpt(idx, add_activations=steering_vector)
        return logits


# %%
from torch.optim import AdamW

probed_model = GPTWithActivationAddition(model, layer=4).to(device)
optimizer = AdamW(probed_model.activation_layers.parameters(), lr=0.01)
EPOCHS = 500
# Training loop
def train_linear_probe():
    for epoch in range(EPOCHS):
        probed_model.activation_layers.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = probed_model(inputs)
            logits_reshaped = outputs.view(-1, 64)  # torch.Size([32768, 64])
            targets_reshaped = targets.view(-1)  # torch.Size([32768])

            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Estimate on test set
        probed_model.activation_layers.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = probed_model(inputs)
                logits_reshaped = outputs.view(-1, 64)
                targets_reshaped = targets.view(-1)
                loss = F.cross_entropy(logits_reshaped, targets_reshaped)
                test_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}")

train_linear_probe()
    
# %%
import numpy as np
from mazegpt.utils import display_maze, parse_maze


for test_id in range(100):
    example, example_target = test_dataset[test_id]
    example_target = example_target.cpu()
    example_row = dataset[len(train_dataset) + test_id]

    maze, directions = parse_maze(example_row["maze"] + ";" + "".join(example_row["directions"]) + ";\n")
    markers = [(marker, pos, correct_move) for marker, pos, correct_move in example_row["markers"] if marker == MARKER_TARGET]

    for marker, pos, correct_move in markers:
        mistake_abs_pos = len(example_row["maze"]) + 1 + pos
        print(f"Mistake at example {test_id} at position {mistake_abs_pos}")

        token_id = mistake_abs_pos - 1

        example, _ = test_dataset[test_id]
        example_row = dataset[len(train_dataset) + test_id]
        # only first tokens
        example_trimmed = example[:token_id]

        maze, directions = parse_maze(decode(example_trimmed.tolist()))
        print(f"The marker says the target is {correct_move}")
        print(f"The target tensor is {decode([example_target[token_id+1].item()])}")
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
        steering_vector[5] = probed_model.activation_layers[0].weight[0]


        with torch.no_grad():  
            example_logits, _ = model(example_trimmed.unsqueeze(0))
            example_logits = example_logits.squeeze(0)
            softmaxed = F.softmax(example_logits, dim=-1).cpu()


        with torch.no_grad():  
            steered_logits, _ = model(example_trimmed.unsqueeze(0), add_activations=steering_vector)
            steered_logits = steered_logits.squeeze(0)
            steered_softmaxed = F.softmax(steered_logits, dim=-1).cpu()

        display_logits(softmaxed[-1], steered_softmaxed[-1])


# %%
