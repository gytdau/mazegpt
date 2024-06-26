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
    CORRECT = "c"
    MISTAKE = "m"


class Markers(Enum):
    DECISION = "decision"
    MISTAKE = "mistake"
    NON_MISTAKE = "non_mistake"
    CORRECTION = "correction"
    FALLIBLE_GOES_SOUTH = "fallible_goes_south"
    FALLIBLE_GOES_NORTH = "fallible_goes_north"
    FALLIBLE_GOES_EAST = "fallible_goes_east"
    FALLIBLE_GOES_WEST = "fallible_goes_west"
    SOUTH_IS_POSSIBLE = "south_is_possible"
    NORTH_IS_POSSIBLE = "north_is_possible"
    EAST_IS_POSSIBLE = "east_is_possible"
    WEST_IS_POSSIBLE = "west_is_possible"


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


def display_maze(
    maze: List[List[Tile]], directions: str = None, signal: List[float] = None
) -> None:
    display(HTML(maze_html(maze, directions, signal)))


def display_mazes_in_grid(
    params: List[Tuple[List[List[Tile]], str, List[float]]], title: str = None
) -> None:
    # 3x3 grid
    if title:
        html = f"<h3>{title}</h3>"
    else:
        html = ""
    html += "<div style='display: grid; grid-template-columns: repeat(3, 1fr); grid-gap: 10px;'>"
    for maze, directions in params:
        html += "<div>"
        html += maze_html(maze, directions)
        html += "</div>"
    html += "</div>"
    display(HTML(html))


def maze_html(
    maze: List[List[Tile]], directions: str = None, signal: List[float] = None
) -> None:
    start_pos = next(
        (
            (i, j)
            for i, row in enumerate(maze)
            for j, tile in enumerate(row)
            if tile == Tile.START
        ),
        None,
    )
    if not start_pos:
        return "Start position not found."

    x, y = start_pos
    path_positions = [
        (start_pos, "start")
    ]  # List of positions (x, y) visited with directions

    direction_mapping = {
        Tile.EAST.value: (0, 1),
        Tile.WEST.value: (0, -1),
        Tile.NORTH.value: (-1, 0),
        Tile.SOUTH.value: (1, 0),
    }

    direction_arrows = {
        Tile.EAST.value: "→",
        Tile.WEST.value: "←",
        Tile.NORTH.value: "↑",
        Tile.SOUTH.value: "↓",
        "start": "",
        "end": "X",
    }

    if directions:
        for move in directions:
            if move in direction_mapping:
                dx, dy = direction_mapping[move]
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
                    path_positions.append(((x, y), move))
                    x, y = nx, ny
                else:
                    break

    path_positions.append(((x, y), "end"))

    html_str = '<div><div style="border: 3px solid grey; display: inline-block; flex-direction: column;"><table style="border-collapse: collapse;">\n'
    for i, row in enumerate(maze):
        html_str += "  <tr>\n"
        for j, tile in enumerate(row):
            background_color = None
            if (i, j) == start_pos:
                background_color = "#4ade80"
            elif tile == Tile.WALL:
                background_color = "#d4d4d8"
            elif tile == Tile.END:
                background_color = "#f87171"
            else:
                background_color = "#f4f4f5"

            cell_style = "background-color: {};".format(background_color)
            cell_content = ""
            for path_i in range(len(path_positions)):
                pos, dir = path_positions[path_i]
                if (i, j) == pos:
                    cell_content += direction_arrows[dir]
                    if signal and path_i < len(signal):
                        cell_style += interpolate_white_to_blue(signal[path_i] * 100)
            html_str += f'    <td style="width: 20px; height: 20px; border: 1px solid; {cell_style} text-align: center; color: black;">{html.escape(cell_content)}</td>\n'
        html_str += "  </tr>\n"
    html_str += "</table></div></div>"

    return html_str


def display_maze_with_markers(
    maze: List[List[Tile]], directions: str = None, markers: List[int] = None
) -> None:
    maze_token_length = len(serialize_maze(maze) + ";")
    for marker, marker_pos, _ in markers:
        print(
            "At marker",
            marker,
            "at move",
            marker_pos,
            " (abs:",
            maze_token_length + marker_pos,
            "tokens)",
        )
        truncated_directions = directions[:marker_pos]
        display_maze(maze, truncated_directions)

    print("Full path")
    display_maze(maze, directions)


def parse_directions(output: str) -> List[str]:
    return output


def parse_maze(output: str) -> Tuple[List[List[Tile]], List[str]]:
    mazes = output.split(";\n")
    maze, directions = mazes[0].split(";")
    maze = maze.split("\n")
    maze = [[Tile(tile) for tile in row] for row in maze]
    return maze, directions


def serialize_maze(maze: List[List[Tile]]) -> str:
    return "\n".join("".join(tile.value for tile in row) for row in maze)


def serialize_sample(maze: List[List[Tile]], directions: List[str]) -> str:
    return serialize_maze(maze) + ";" + "".join(directions)
