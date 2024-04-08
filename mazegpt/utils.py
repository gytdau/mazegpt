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


def display_maze(maze: List[List[Tile]], directions: str = None) -> None:
    start_pos = next(((i, j) for i, row in enumerate(maze) for j, tile in enumerate(row) if tile == Tile.START), None)
    if not start_pos:
        return "Start position not found."

    x, y = start_pos
    path_positions = [(start_pos, None)]  # List of positions (x, y) visited with directions

    direction_mapping = {
        "E": (0, 1),
        "W": (0, -1),
        "N": (-1, 0),
        "S": (1, 0)
    }

    direction_arrows = {
        "E": "→",
        "W": "←",
        "N": "↑",
        "S": "↓"
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
            for pos, dir in path_positions:
                if (i, j) == pos and dir:
                    cell_content += direction_arrows[dir]
            html_str += f'    <td style="width: 20px; height: 20px; border: 1px solid; {cell_style} text-align: center; color: black;">{html.escape(cell_content)}</td>\n'
        html_str += '  </tr>\n'
    html_str += '</table>'

    display(HTML(html_str))

def display_maze_with_markers(maze: List[List[Tile]], directions: str = None, markers: List[int] = None) -> None:
    for marker, marker_pos in markers:
        print("At marker", marker, "at position", marker_pos)
        truncated_directions = directions[:marker_pos]
        display_maze(maze, truncated_directions)
    
    print("Full path")
    display_maze(maze, directions)

def parse_directions(output: str) -> List[str]:
    return output


def parse_maze(output: str) -> Tuple[List[List[Tile]], List[str]]:
    maze, directions = output.split(";")
    maze = maze.split("\n")
    return maze, directions

