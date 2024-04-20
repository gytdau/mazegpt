
# %% [markdown]
# Convenience file to sample the solution to a maze quickly.
from gytis import imports
from mazegpt.sample import model, encode, decode, device, stoi
import torch
from mazegpt.utils import display_maze, parse_maze

prompt="""##########
#   #    e
### # ### 
#   # #s# 
# ### # # 
# #   # # 
# # ### # 
# #     # 
# ####### 
#         ;"""


prompt_size = len(prompt)
encoded = encode(prompt)
encoded = torch.tensor(encoded).unsqueeze(0).to(device)
completion_tokens = model.generate(encoded, max_new_tokens=100)
completion = decode(completion_tokens.squeeze(0)[prompt_size:].tolist())
# cut off at first ;
seperator = ";"
completion = completion[:completion.index(seperator)].strip()
maze, directions = parse_maze(prompt + completion)
display_maze(maze, directions)

# %%
import torch.nn.functional as F

token_id = len(prompt + completion) - 0

with torch.no_grad():  
    example_logits, _ = model(completion_tokens[:, :token_id])
    example_logits = example_logits.squeeze(0).squeeze(0)
    # softmax
    softmaxed = F.softmax(example_logits, dim=-1).cpu()

for i in stoi.values():
    print(f"{repr(decode([i]))}: {softmaxed[i]:.2f}")

# %%
