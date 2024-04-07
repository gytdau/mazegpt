# %%
output = """###############
#s#   #       #
# # # # # #####
#     #       #
####### ### # #
# #         # #
####### ##### #
# #   # #   # #
# ####### ### #
#   #     # # #
##### ####### #
#   #       # #
### ######### #
#            e#
############### EEEESSWWSSSSEENNEESSWWSSSSEEEEEESSEEEEEE
"""


# %%
from enum import Enum
from typing import List

class Tile(Enum):
    EMPTY = " "
    WALL = "#"
    START = "s"
    END = "e"
    PATH = "."

def render_path_on_maze_with_directions(maze: List[List[Tile]], directions: List[str]) -> None:
    # Find the start position
    start_x, start_y = None, None
    for i, row in enumerate(maze):
        for j, _ in enumerate(row):
            if maze[i][j] == Tile.START:
                start_x, start_y = i, j
                break
        if start_x is not None:
            break

    # Apply directions to mark the path
    x, y = start_x, start_y
    for direction in directions:
        if direction == "N":
            x -= 1
        elif direction == "S":
            x += 1
        elif direction == "E":
            y += 1
        elif direction == "W":
            y -= 1
        # Mark the path by setting the current position to a special path tile, if it's not start or end
        if 0 <= x < len(maze) and 0 <= y < len(maze[0]):
            if maze[x][y] not in [Tile.START, Tile.END]:
                # if within range:
                    maze[x][y] = Tile.PATH  # Consider using a different tile or mechanism to mark the path if necessary

    # Print the maze with the path
    for i, row in enumerate(maze):
        for j, tile in enumerate(row):
            if (i, j) == (start_x, start_y):
                print('S', end='')
            elif tile == Tile.END:
                print('E', end='')
            elif tile == Tile.PATH:
                print('.', end='')  # Mark path
            else:
                print(tile.value, end='')
        print()

def parse_maze(output: str) -> List[List[Tile]]:
    # find first capital letter
    first_capital_letter = next((c for c in output if c.isupper()), None)
    # find the index of the first capital letter
    first_capital_index = output.index(first_capital_letter)
    # split the output at the first capital letter
    maze_output, directions_output = output[:first_capital_index], output[first_capital_index:]
    parsed_maze = [[Tile(c) for c in row] for row in maze_output.split("\n")[:-1]]
    parsed_directions = parse_directions(directions_output)
    print(parsed_maze)
    print(parsed_directions)
    return parsed_maze, parsed_directions


def parse_directions(output: str) -> List[str]:
    return output.split("\n")[-1]

maze, directions = parse_maze(output)

# %%
render_path_on_maze_with_directions(maze, directions)

# %%
