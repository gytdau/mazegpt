# %%
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from gytis import imports
from mazegpt.model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 2  # number of samples to draw
max_new_tokens = 1000  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
# exec(open("configurator.py").read())  # overrides from command line or config file
exec(open("config/train_maze.py").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# %%
start = """###############
#s  #         #
### ### # #####
# #   # #     #
# ### ##### # #
#     #   # #
# ##### # ### #
#       # #   #
######### # # #
#       #   # #
# ### ####### #
#   #       # #
### ####### # #
#         #  e#
###############"""

# encode the beginning of the prompt
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# # run generation
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print("---------------")
# %% [markdown]
# ## Linear Probes
# We're training a probe to predict whether the current move is a decision.
# %%
import torch.nn as nn
import json
from mazegpt.utils import display_maze, display_maze_with_markers, parse_maze
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import cProfile

class GPTWithLinearProbe(nn.Module):
    def __init__(self, gpt_model, layer, num_classes):
        super().__init__()
        self.gpt = gpt_model
        self.layer = layer
        self.gpt.eval()  # Freeze the GPT model weights
        self.linear_probe = nn.Linear(gpt_model.config.n_embd, num_classes).to(device)
    
    def forward(self, idx):
        hidden_states = self.gpt(idx, return_hidden_states=True)
        hidden_state = hidden_states[self.layer]  
        logits = self.linear_probe(hidden_state)
        return logits

dataset_path = os.path.join(os.path.dirname(__file__), "data/mazes/data.jsonl")

num_samples_for_probe = 10000
with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f.readlines()[:num_samples_for_probe]]

marker_target = "decision"
inputs = []
targets = []
for i, row in enumerate(dataset):
    serialized = row["maze"] + ";" + "".join(row["directions"]) + ";\n"
    tokens = torch.tensor(encode(serialized))  # Ensure this is a tensor or convert it into one
    classes = torch.zeros(len(tokens), dtype=torch.long)

    marker_positions = [len(row["maze"]) + 1 + marker_pos for marker, marker_pos in row["markers"] if marker == marker_target]
    classes[marker_positions] = 1
    
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

# %% [markdown]
# ## Example target
test_id = 4
example, example_target = test_dataset[test_id]
example_target = example_target.cpu()
example_row = dataset[len(train_dataset) + test_id]

maze, directions = parse_maze(example_row["maze"] + ";" + "".join(example_row["directions"]) + ";\n")
seperator = encode(";")[0]
maze_end, directions_end = [v.item() for v in (example == seperator).nonzero()]
target_signal = example_target[maze_end+1:directions_end].cpu()
display_maze(maze, directions, signal=list(target_signal))



# %%

# Model
probed_model = GPTWithLinearProbe(model, layer=4, num_classes=2)
optimizer = Adam(probed_model.parameters(), lr=0.001)

# %%

# Training loop
def train_linear_probe():
    epochs = 1
    for epoch in range(epochs):
        probed_model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = probed_model(inputs)
            outputs = outputs.transpose(1, 2)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Estimate on test set
        probed_model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = probed_model(inputs)
                outputs = outputs.transpose(1, 2)
                loss = F.cross_entropy(outputs, targets)
                test_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataset)}, Test Loss: {test_loss/len(test_dataset)}")
    

train_linear_probe()
# %%
# Plot the hidden state of an arbitrary input.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mazegpt.utils import display_maze, display_maze_with_markers, parse_maze

from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter
import plotly.graph_objects as go

# Ensure the model is in evaluation mode
probed_model.eval()


test_id = 4
example, example_target = test_dataset[test_id]
example_target = example_target.cpu()
example_row = dataset[len(train_dataset) + test_id]

# Get the model's predictions for this example
with torch.no_grad():  # No need to track gradients here
    example_logits = probed_model(example.unsqueeze(0)).squeeze(0)
    softmaxed = F.softmax(example_logits, dim=-1).cpu()

decision = softmaxed[:, 1]
not_decision = softmaxed[:, 0]

# Generate x values representing each token position
token_positions = list(range(example.size(0)))

# Create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=token_positions, y=decision, mode='lines+markers',
                         name='decision', marker=dict(size=5), line=dict(shape='spline')))
# Ground truth
fig.add_trace(go.Scatter(x=token_positions, y=example_target, mode='lines+markers',
                         name='Ground Truth', marker=dict(size=5), line=dict(shape='spline')))

fig.update_layout(title=f'Signal from Linear Probe for Class {1} on Example {0}',
                    xaxis_title='Token Position',
                    yaxis_title='Probability',
                    template='plotly_dark')
fig.show()

result_signal = list(decision[maze_end+1:directions_end].cpu().tolist())
display_maze(maze, directions, signal=result_signal)


# %%
