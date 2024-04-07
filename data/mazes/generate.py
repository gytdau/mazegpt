# %%
from enum import Enum
from typing import List, Tuple
import random

class Tile(Enum):
    EMPTY = " "
    WALL = "#"
    START = "S"
    END = "E"

def generate_maze(n: int) -> List[List[Tile]]:
    # Initialize maze with all walls
    maze = [[Tile.WALL for _ in range(n)] for _ in range(n)]

    # Starting point
    start = (1, 1)
    maze[start[0]][start[1]] = Tile.START
    stack = [start]

    # Directions: up, right, down, left
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

    while stack:
        current = stack[-1]
        x, y = current
        # Check possible directions
        possible_directions = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] == Tile.WALL:
                possible_directions.append((dx, dy))
        if possible_directions:
            dx, dy = random.choice(possible_directions)
            next_x, next_y = x + dx, y + dy
            # Mark the path and the next cell
            maze[next_x][next_y] = Tile.EMPTY
            maze[x + dx//2][y + dy//2] = Tile.EMPTY
            stack.append((next_x, next_y))
        else:
            # No way forward: backtrack
            stack.pop()

    # Set an end point
    end_found = False
    for row in range(n-2, 0, -1):
        for col in range(n-2, 0, -1):
            if maze[row][col] == Tile.EMPTY:
                maze[row][col] = Tile.END
                end_found = True
                break
        if end_found:
            break

    return maze

def display_maze(maze: List[List[Tile]]) -> None:
    for row in maze:
        print(''.join(tile.value for tile in row))

# Example usage
n = 15  # Size of the maze
maze = generate_maze(n)
display_maze(maze)

# %%
