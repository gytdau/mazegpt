# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from datasets import load_dataset, Dataset

# Define the model configuration
config = GPT2Config(
    vocab_size=64,
    n_positions=512,
    n_ctx=512,
    n_embd=384,
    n_layer=6,
    n_head=6,
    activation_function='gelu',
    resid_pdrop=0.2,
    embd_pdrop=0.2,
    attn_pdrop=0.2,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
)

# Load in the model from the checkpoint from safetensors
model = GPT2LMHeadModel.from_pretrained("results/checkpoint-30000")


# Load the tokenizer
tokenizer = Tokenizer(BPE(
    vocab="./model/vocab.json",
    merges="./model/merges.txt",
))
tokenizer.save("./model/tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./model/tokenizer.json",
                                    bos_token="[BOS]",
                                    eos_token="[EOS]",
                                    unk_token="[UNK]",
                                    pad_token="[PAD]",
                                    mask_token="[MASK]",
                                    )

# %%


device = "cuda"
model.to(device)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

prompt = "#"
generated_text = generate_text(prompt)
print(generated_text)

# %%
