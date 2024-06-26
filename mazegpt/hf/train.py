# %%
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

# Initialize the model from scratch
model = GPT2LMHeadModel(config)

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

# import json
# text_data = ""
# with open('../data/mazes/fallible/data.jsonl', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         parsed = json.loads(line)
#         text_data += parsed["maze"] + ";" + parsed["directions"] + ";\n\n"

# # Write text_data to a file.
# with open('../data/mazes/fallible/text_data.txt', 'w') as f:
#     f.write(text_data)

# %%

dataset = load_dataset("text", data_files=["../data/mazes/fallible/text_data.txt"], keep_in_memory=True, sample_by="paragraph")

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# %%
# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset into train and validation sets
split = tokenized_dataset["train"].train_test_split(test_size=0.0025)
train_dataset = split["train"]
val_dataset = split["test"]

# %%
# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=10000,
    eval_steps=10000,
    logging_steps=500,
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
device = "cuda"

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
