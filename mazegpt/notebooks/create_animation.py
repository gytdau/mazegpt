
# %% [markdown]
# Convenience file to sample the solution to a maze quickly.
from gytis import imports
from mazegpt.sample import model, encode, decode, device, stoi
import torch
from mazegpt.utils import display_maze, parse_maze


prompt="""     # e 
 ### ### 
 #s  #   
 ##### # 
 #     # 
 ##### # 
 #   # # 
 # # # # 
   #   # ;"""
path = "EENNWWWWSSSSSSSSEENNEESSEENNNNWENNEENNW"

# display_maze(*parse_maze(prompt + path))

import plotly.graph_objects as go

char_to_arrow = {
    "N": "↑",
    "E": "→",
    "S": "↓",
    "W": "←",
}

directions = "NESW"

path_range = [20, 21] #[20, 40]
for path_pos in range(path_range[0], path_range[1]):
    path_suffix = path[:path_pos]
    total_prompt = prompt + path_suffix

    # display_maze(*parse_maze(total_prompt))


    prompt_size = len(total_prompt)
    encoded = encode(total_prompt)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)
    completion_tokens = model.generate(encoded, max_new_tokens=120, temperature=0.1)
    completion = decode(completion_tokens.squeeze(0)[prompt_size:].tolist())
    # cut off at first ;
    seperator = ";"
    completion = completion[:completion.index(seperator)].strip()
    maze, directions = parse_maze(prompt + completion)
    # display_maze(maze, directions)

    import torch.nn.functional as F


    token_id = len(total_prompt)

    # Display up to token_id
    trimmed = completion_tokens[:, :token_id]
    trimmed = trimmed.squeeze(0)
    trimmed = trimmed.tolist()
    trimmed = decode(trimmed)


    with torch.no_grad():  
        example_logits, _ = model(completion_tokens)
        example_logits = example_logits.squeeze(0).squeeze(0)
        # softmax
        softmaxed = F.softmax(example_logits, dim=-1).cpu()[token_id-1]

    probabilities = [softmaxed[stoi[char]].item() for char in directions]
    arrows = [char_to_arrow[char] for char in directions]
    fig = go.Figure(go.Bar(
        x=probabilities,
        y=arrows,
        orientation='h',
    ))

    fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title='Next move prob',
        yaxis=dict(
            tickfont=dict(
                family='Arial',
                size=16,
                color='black'
            ),
            tickmode='array',
            tickvals=list(range(len(arrows))),
            ticktext=arrows,
        ),
        xaxis=dict(
                    range=[0, 1],
        ),
        margin=dict(
            pad=10
        ),
        width=350,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    # save to png
    fig.show()

# %%
