import torch.nn as nn

class GPTWithProbe(nn.Module):
    def __init__(self, gpt_model, layer, num_classes):
        super().__init__()
        self.gpt = gpt_model
        self.layer = layer
        self.gpt.eval()  # Freeze the GPT model weights
        self.probe_layers = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, idx):
        _, _, hidden_states = self.gpt(idx, return_hidden_states=True)
        hidden_state = hidden_states[self.layer]  
        logits = self.probe_layers(hidden_state)
        return logits


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

