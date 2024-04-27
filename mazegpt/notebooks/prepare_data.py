
from mazegpt.sample import  encode, device
import os
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

def prepare_data(path, num_samples, marker_target):
    dataset_path = os.path.join(os.path.dirname(__file__), path)

    num_samples_for_probe = num_samples
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()[:num_samples_for_probe]]
    
    def get_ground_truth(tokens, row):
        classes = torch.zeros(len(tokens), dtype=torch.long)

        offset = -1
        abs_pos_offset = len(row["mistake_prob"] + ";" + row["maze"] + ";") + offset

        marker_positions = [abs_pos_offset + marker_pos for marker, marker_pos, _ in row["markers"] if marker == marker_target]
        classes[marker_positions] = 1

        return classes



    inputs = []
    targets = []
    for i, row in enumerate(dataset):
        serialized = row["mistake_prob"] + ";" + row["maze"] + ";" + "".join(row["directions"]) + ";\n"
        tokens = torch.tensor(encode(serialized))  # Ensure this is a tensor or convert it into one

        classes = get_ground_truth(tokens, row)

        if classes.sum() == 0:
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

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"There are {len(padded_inputs)} samples in the dataset")
    return dataset, train_loader, test_loader, get_ground_truth

