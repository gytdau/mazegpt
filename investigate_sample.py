# %%
output = """###############
#s  #         #
### ### # #####
# #   # #     #
# ### ##### # #
#     #   # # #
# ##### # ### #
#       # #   #
######### # # #
#       #   # #
# ### ####### #
#   #       # #
### ####### # #
#         #  e#
############### EESSEESSWWWWSSEEEEEENNEESSSSWWWWWWSSEEEENNEESSEE
"""


# %%
from enum import Enum
from typing import List
import html

class Tile(Enum):
    EMPTY = " "
    WALL = "#"
    START = "s"
    END = "e"
    PATH = "."

def create_maze_html(maze: List[List[Tile]], directions: str) -> str:
    # Starting position
    start_pos = next(((i, j) for i, row in enumerate(maze) for j, tile in enumerate(row) if tile == Tile.START), None)
    if not start_pos:
        return "Start position not found."
    
    x, y = start_pos
    path_positions = [start_pos]  # List of positions (x, y) visited

    # Mapping of directions to movement deltas
    direction_mapping = {
        "E": (0, 1),
        "W": (0, -1),
        "N": (-1, 0),
        "S": (1, 0)
    }

    for move in directions:
        if move in direction_mapping:
            dx, dy = direction_mapping[move]
            x += dx
            y += dy
            # Check bounds and wall collision
            if 0 <= x < len(maze) and 0 <= y < len(maze[0]):
                path_positions.append((x, y))
            else:
                # If out of bounds, stop processing further moves
                break

    # Create HTML representation
    html_str = '<table style="border-collapse: collapse;">\n'
    for i, row in enumerate(maze):
        html_str += '  <tr>\n'
        for j, tile in enumerate(row):
            color = ""
            if (i, j) == start_pos:
                color = "green"
            elif (i, j) in path_positions:
                color = "#0000FF"
            elif tile == Tile.WALL:
                color = "black"
            elif tile == Tile.END:
                color = "red"
            cell = html.escape(tile.value) if tile != Tile.PATH or (i, j) in path_positions else " "
            html_str += f'    <td style="width: 20px; height: 20px; border: 1px solid; background-color: {color}; text-align: center;">{cell}</td>\n'
        html_str += '  </tr>\n'
    html_str += '</table>'

    return html_str

def parse_maze(output: str) -> List[List[Tile]]:
    # find first capital letter
    first_capital_letter = next((c for c in output if c.isupper()), None)
    # find the index of the first capital letter
    first_capital_index = output.index(first_capital_letter)
    # split the output at the first capital letter
    maze_output, directions_output = output[:first_capital_index], output[first_capital_index:]
    parsed_maze = [[Tile(c) for c in row] for row in maze_output.split("\n")]
    parsed_directions = parse_directions(directions_output)
    print(parsed_maze)
    print(parsed_directions)
    return parsed_maze, parsed_directions


def parse_directions(output: str) -> List[str]:
    return output

maze, directions = parse_maze(output)

# %%
from IPython.display import display, HTML
html_str = create_maze_html(maze, directions)

display(HTML(html_str))

# %%
