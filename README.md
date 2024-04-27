# A Toy Model Of Fallible Maze Solvers

This repo tests the ability of language models to solve mazes and to correct their errors.

You can see the experiments run in `mazegpt/notebooks`:
- `mazegpt/notebooks/activation_addition.py`
- `mazegpt/notebooks/investigate_sample.py`
- `mazegpt/notebooks/linear_probes.py`


This repo was originally forked from nanogpt.

## Training and sampling

1. Generate the dataset via `python3 data/mazes/generate.py`
2. Prepare it into train and test sets via `python3 data/mazes/prepare.py`
3. To train:
    ```
python3 train.py config/train_maze_correctable.py
python3 train.py config/train_maze_oracle.py
    ```
4. To sample:

    ```
python3 sample.py
    ```

