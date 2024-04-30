from torch.utils.data import Dataset
from transformers import ByteLevelBPETokenizer, BertProcessing
from pathlib import Path
import torch

class MazeDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "./model/vocab.json",
            "./model/merges.txt",
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        for src_file in src_files:
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])