# A Toy Model Of Fallible Maze Solvers

This repo tests the ability of language models to solve mazes and to correct their errors. You can see most 

This repo was originally forked from nanogpt.


## Training and sampling

To train:

```
python3 train.py config/train_maze_correctable.py
python3 train.py config/train_maze_oracle.py
```

To sample:

```
python3 sample.py
```

