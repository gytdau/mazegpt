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

from collections import deque

def find_shortest_path(maze: List[List[Tile]]) -> List[Tuple[int, int]]:
    n = len(maze)
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = [[False for _ in range(n)] for _ in range(n)]
    distance = [[0 for _ in range(n)] for _ in range(n)]
    prev = [[None for _ in range(n)] for _ in range(n)]

    # Find start
    start = end = None
    for i in range(n):
        for j in range(n):
            if maze[i][j] == Tile.START:
                start = (i, j)
            elif maze[i][j] == Tile.END:
                end = (i, j)
    
    if not start or not end:
        return []  # No start or end
    
    queue = deque([start])
    visited[start[0]][start[1]] = True

    # BFS
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break  # Found the end
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny] and maze[nx][ny] != Tile.WALL:
                queue.append((nx, ny))
                visited[nx][ny] = True
                prev[nx][ny] = (x, y)
                distance[nx][ny] = distance[x][y] + 1

    # Reconstruct path
    path = []
    at = end
    while at:
        path.append(at)
        at = prev[at[0]][at[1]]
    path.reverse()

    return path if path[0] == start else []  # Return path if start is included, else return empty

# Example usage
path = find_shortest_path(maze)
print("Path from start to end:")
for step in path:
    print(step)

# %%

def render_path_on_maze(maze: List[List[Tile]], path: List[Tuple[int, int]]) -> None:
    # Convert path to a set for efficient lookup
    path_set = set(path)

    for i, row in enumerate(maze):
        for j, tile in enumerate(row):
            if (i, j) in path_set and tile == Tile.EMPTY:
                print('.', end='')
            elif tile == Tile.START:
                print('S', end='')
            elif tile == Tile.END:
                print('E', end='')
            else:
                print(tile.value, end='')
        print()  # Newline after each row

# Using the existing maze and path from the previous example
render_path_on_maze(maze, path)

# %%
