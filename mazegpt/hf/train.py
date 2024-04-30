# %%
from transformers import GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from datasets import load_dataset, Dataset

# Define the model configuration
config = GPT2Config(
    vocab_size=64,
    n_positions=512,
    n_ctx=512,
    n_embd=384,
    n_layer=2,
    n_head=6,
    activation_function='gelu',
    resid_pdrop=0.2,
    embd_pdrop=0.2,
    attn_pdrop=0.2,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
)

# Initialize the model from scratch
model = GPT2LMHeadModel(config)

# Load the tokenizer
tokenizer = Tokenizer(BPE(
    vocab="./model/vocab.json",
    merges="./model/merges.txt",
))

# %%
# %%

# import json
# text_data = ""
# with open('../data/mazes/fallible/data.jsonl', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         parsed = json.loads(line)
#         text_data += parsed["maze"] + ";" + parsed["directions"] + ";\n"

# # Write text_data to a file.
# with open('text_data.txt', 'w') as f:
#     f.write(text_data)

# %%

dataset = load_dataset("text", data_files=["../data/mazes/fallible/text_data.txt"])
def tokenize_function(examples):
    return {"input_ids": [enc.ids for enc in tokenizer.encode_batch(examples["text"])]}

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# %%
# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset into train and validation sets
split = tokenized_dataset["train"].train_test_split(test_size=0.2)
train_dataset = split["train"]
val_dataset = split["test"]

# %%
# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Start the training process
trainer.train()
# %%
