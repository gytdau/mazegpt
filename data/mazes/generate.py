# %%
from enum import Enum
from typing import List, Tuple
import random



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


from typing import List


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
        if maze[x][y] not in [Tile.START, Tile.END]:
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

render_path_on_maze_with_directions(maze, path)

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
