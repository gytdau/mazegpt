# %%
import numpy as np
import os
import pickle
import json
from mazegpt.utils import Tile
# Assign ints for all
stoi = {}

new_tokens = [tile.value for tile in Tile]
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

        serialized_line = maze
        serialized_line += Tile.SEPERATOR.value + directions + Tile.SEPERATOR.value + Tile.NEW_LINE.value

        for char in serialized_line:
            tokens.append(stoi[char])

    return tokens


def convert_log():
    with open("data.jsonl", "r") as f:
        encoded = encode(f.readlines())

        train = encoded[: int(len(encoded) * 0.9)]
        val = encoded[int(len(encoded) * 0.9) :]

        train_ids = np.array(train, dtype=np.uint16)
        val_ids = np.array(val, dtype=np.uint16)
        train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
        val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    convert_log()

# %%
