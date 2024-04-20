from enum import Enum
import html
from typing import List, Tuple

from IPython.display import display, HTML


class Tile(Enum):
    EMPTY = " "
    WALL = "#"
    START = "s"
    END = "e"
    PATH = "."
    SEPERATOR = ";"
    SOUTH = "S"
    NORTH = "N"
    EAST = "E"
    WEST = "W"
    NEW_LINE = "\n"

class Markers(Enum):
    DECISION = "decision"
    MISTAKE = "mistake"
    NON_MISTAKE = "non_mistake"
    CORRECTION = "correction"


def interpolate_white_to_blue(p):
    """
    Interpolate from white to blue based on a percentage.

    Parameters:
    p (float): The percentage (0 to 100) for interpolation.

    Returns:
    str: The CSS RGB value as a string.
    """
    # Ensure p is within the bounds [0, 100]
    p = max(0, min(100, p))
    
    # Calculate the R and G values (B is always 255)
    R = G = int(255 * (1 - p / 100))
    B = 255

    # Return the CSS RGB value
    return f"background-color: rgb({R}, {G}, {B});"

def display_maze(maze: List[List[Tile]], directions: str = None, signal: List[float] = None) -> None:
    start_pos = next(((i, j) for i, row in enumerate(maze) for j, tile in enumerate(row) if tile == Tile.START), None)
    if not start_pos:
        return "Start position not found."

    x, y = start_pos
    path_positions = [(start_pos, None)]  # List of positions (x, y) visited with directions

    direction_mapping = {
        Tile.EAST.value: (0, 1),
        Tile.WEST.value: (0, -1),
        Tile.NORTH.value: (-1, 0),
        Tile.SOUTH.value: (1, 0)
    }

    direction_arrows = {
        Tile.EAST.value: "→",
        Tile.WEST.value: "←",
        Tile.NORTH.value: "↑",
        Tile.SOUTH.value: "↓"
    }

    if directions:
        for move in directions:
            if move in direction_mapping:
                dx, dy = direction_mapping[move]
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
                    x, y = nx, ny
                    path_positions.append(((x, y), move))
                else:
                    break

    html_str = '<table style="border-collapse: collapse;">\n'
    for i, row in enumerate(maze):
        html_str += '  <tr>\n'
        for j, tile in enumerate(row):
            cell_style = 'background-color: {};'.format("lightgreen" if (i, j) == start_pos else "lightgray" if tile == Tile.WALL else "tomato" if tile == Tile.END else "white")
            cell_content = ""
            for path_i in range(len(path_positions)):
                pos, dir = path_positions[path_i]
                if (i, j) == pos and dir:
                    cell_content += direction_arrows[dir]
                    if signal and path_i < len(signal):
                        cell_style += interpolate_white_to_blue(signal[path_i] * 100)
            html_str += f'    <td style="width: 20px; height: 20px; border: 1px solid; {cell_style} text-align: center; color: black;">{html.escape(cell_content)}</td>\n'
        html_str += '  </tr>\n'
    html_str += '</table>'

    display(HTML(html_str))

def display_maze_with_markers(maze: List[List[Tile]], directions: str = None, markers: List[int] = None) -> None:
    maze_token_length = len(serialize_maze(maze) + ";")
    for marker, marker_pos, _ in markers:
        print("At marker", marker, "at move", marker_pos, " (abs:", maze_token_length + marker_pos, "tokens)")
        truncated_directions = directions[:marker_pos]
        display_maze(maze, truncated_directions)
    
    print("Full path")
    display_maze(maze, directions)

def parse_directions(output: str) -> List[str]:
    return output


def parse_maze(output: str) -> Tuple[List[List[Tile]], List[str]]:
    mazes = output.split(';\n')
    maze, directions = mazes[0].split(";")
    maze = maze.split("\n")
    maze = [[Tile(tile) for tile in row] for row in maze]
    return maze, directions

def serialize_maze(maze: List[List[Tile]]) -> str:
    return "\n".join("".join(tile.value for tile in row) for row in maze)

def serialize_sample(maze: List[List[Tile]], directions: List[str]) -> str:
    return serialize_maze(maze) + ";" + "".join(directions)

