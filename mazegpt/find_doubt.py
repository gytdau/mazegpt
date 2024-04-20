# %%
"""
Sample from a trained model
"""
from gytis import imports
from mazegpt.sample import model, device, encode, decode, stoi
from mazegpt.utils import display_maze, display_maze_with_markers, parse_maze
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset_path = os.path.join(os.path.dirname(__file__), "data/mazes/correctable/data.jsonl")

num_samples_for_probe = 50_000
with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f.readlines()[:num_samples_for_probe]]

marker_target = "mistake"
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

# %%
THRESHOLD = 0.05
for test_id in range(1900):
    example, example_target = test_dataset[test_id]
    example_row = dataset[len(train_dataset) + test_id]
    markers = example_row["markers"] # e.g. [['decision', 10], ['non_mistake', 11]]
    decisions = [marker for marker in markers if marker[0] == marker_target]

    for decision in decisions:
        decision_pos = decision[1]
        token_id = len(example_row["maze"]) + decision_pos
        # only first tokens
        example_trimmed = example[:token_id]


        with torch.no_grad():  
            example_logits, _ = model(example_trimmed.unsqueeze(0))
            example_logits = example_logits.squeeze(0).squeeze(0)
            # softmax
            softmaxed = F.softmax(example_logits, dim=-1).cpu()
        
        possible_tokens = 0
        for i in stoi.values():
            if softmaxed[i] > THRESHOLD:
                possible_tokens += 1
        
        if possible_tokens > 1:
            print(f"Example {test_id}")
            maze, directions = parse_maze(decode(example_trimmed.tolist()))
            display_maze(maze, directions)

            for i in stoi.values():
                if softmaxed[i] > THRESHOLD:
                    print(f"{repr(decode([i]))}: {softmaxed[i]:.2f}")
            print("---")

# %%

THRESHOLD = 0.01
test_id = 1721
example, example_target = test_dataset[test_id]
example_row = dataset[len(train_dataset) + test_id]
markers = example_row["markers"] # e.g. [['decision', 10], ['non_mistake', 11]]
decisions = [marker for marker in markers if marker[0] == marker_target]

for decision in decisions:
    decision_pos = decision[1]
    token_id = len(example_row["maze"]) + decision_pos
    # only first tokens
    example_trimmed = example[:token_id]
    add_token = torch.tensor(encode("S")[0]).unsqueeze(0).to(device)
    example_trimmed = torch.cat((example_trimmed, add_token))

    maze, directions = parse_maze(decode(example_trimmed.tolist()))
    display_maze(maze, directions)


    with torch.no_grad():  
        example_logits, _ = model(example_trimmed.unsqueeze(0))
        example_logits = example_logits.squeeze(0).squeeze(0)
        # softmax
        softmaxed = F.softmax(example_logits, dim=-1).cpu()
    
    print(f"Probabilites for the next token (>{THRESHOLD})")
    for i in stoi.values():
        if softmaxed[i] > THRESHOLD:
            print(f"{repr(decode([i]))}: {softmaxed[i]:.2f}")
## %%

# %%
