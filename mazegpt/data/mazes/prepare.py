# %%
import numpy as np
import os
import pickle
import json
from mazegpt.utils import Tile
# Assign ints for all
stoi = {}

new_tokens = [str(i) for i in range(10)]
new_tokens += [tile.value for tile in Tile]
for token in new_tokens:
    stoi[token] = len(stoi)

vocab_size = len(stoi.items())

itos = {idx: pos for pos, idx in stoi.items()}


def encode(data):
    tokens = []
    for line in data:
        line = json.loads(line)
        maze = line["maze"]
        directions = line["directions"]
        mistake_prob = line["mistake_prob"]

        serialized_line = [mistake_prob, Tile.SEPERATOR.value, maze, Tile.SEPERATOR.value, directions, Tile.SEPERATOR.value, Tile.NEW_LINE.value]
        serialized_line = "".join(serialized_line)

        for char in serialized_line:
            tokens.append(stoi[char])
        

    return tokens


def convert_log(dataset_path):
    with open(f"{dataset_path}/data.jsonl", "r") as f:
        encoded = encode(f.readlines())

        train = encoded[: int(len(encoded) * 0.9)]
        val = encoded[int(len(encoded) * 0.9) :]

        train_ids = np.array(train, dtype=np.uint16)
        val_ids = np.array(val, dtype=np.uint16)
        train_ids.tofile(f"{dataset_path}/train.bin")
        val_ids.tofile(f"{dataset_path}/val.bin")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    with open(f"{dataset_path}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    # convert_log(os.path.join(os.path.dirname(__file__), "oracle"))
    convert_log(os.path.join(os.path.dirname(__file__), "correctable"))

# %%
