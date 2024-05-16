# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import datetime

out_dir = "out-mazes-fallible2"
eval_interval = 500  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_project = "mazes"
wandb_run_name = "fallible-" + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")

dataset = "mazes/fallible"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512  # context of up to 512 previous characters

# baby GPT model :)
n_layer = 10
n_head = 8
n_embd = 384
dropout = 0.2

learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = 15000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

wandb_log = True
compile = False

start = "FILE:prompt.txt"
