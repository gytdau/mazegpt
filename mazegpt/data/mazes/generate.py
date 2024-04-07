# %%
from enum import Enum
from typing import List, Tuple
import random
from gytis import imports
from mazegpt.utils import Tile, display_maze



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


# Example usage
n = 15  # Size of the maze
maze = generate_maze(n)
display_maze(maze)


from typing import List


# %%

from collections import deque

def find_shortest_path_with_directions(maze: List[List[Tile]]) -> List[str]:
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


    # Convert path to directions
    directions = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if dx == 1:
            directions.append("S")  # South
        elif dx == -1:
            directions.append("N")  # North
        elif dy == 1:
            directions.append("E")  # East
        elif dy == -1:
            directions.append("W")  # West

    return directions if path else []  # Return directions if path is valid


# Example usage
path = find_shortest_path_with_directions(maze)
print("Path from start to end:")
print(path)

display_maze(maze, path)


# %%
import json
import tqdm


# %%

import cProfile
import pstats

# Assume your functions are defined here

def main():
    quantity = 1_000_000

    file_path = "data.jsonl"
    with open(file_path, 'a') as file:
        for _ in tqdm.tqdm(range(quantity)):
            maze = generate_maze(15)
            directions = find_shortest_path_with_directions(maze)
            # Convert the maze to a JSON-compatible format
            maze_str = "\\n".join(["".join([tile.value for tile in row]) for row in maze])
            directions_str = "".join(directions)

            # Use string interpolation to create the JSON string manually
            record_json = f'{{"maze": "{maze_str}", "directions": "{directions_str}"}}'

            # Write the JSON string to a file, appending a newline to form the JSON Lines format
            file.write(record_json + '\n')

main()

# %%
