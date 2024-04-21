# %%
from enum import Enum
import os
from typing import List, Tuple
import random
from gytis import imports
from mazegpt.utils import Markers, Tile, display_maze, display_maze_with_markers, serialize_sample



def generate_maze() -> List[List[Tile]]:
    n = random.randint(4, 9)
    # Initialize maze with all walls
    maze = [[Tile.WALL for _ in range(n)] for _ in range(n)]

    # Starting point
    start = (random.randint(0, n-1), random.randint(0, n-1))
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
    empty_cells = [(row, col) for row in range(n) for col in range(n) if maze[row][col] == Tile.EMPTY]
    end_cell = random.choice(empty_cells)
    maze[end_cell[0]][end_cell[1]] = Tile.END
    return maze



maze = generate_maze()
display_maze(maze)




# %%

from collections import deque

MISTAKE_PROBABILITY = 0 #0.2
def find_shortest_path_with_directions(maze: List[List[Tile]]) -> List[str]:
    n = len(maze)
    # Directions: up, right, down, left
    move_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = set()
    distance = [[0 for _ in range(n)] for _ in range(n)]
    prev = [[None for _ in range(n)] for _ in range(n)]
    decision_point = set()

    # Find start
    start = end = None
    for i in range(n):
        for j in range(n):
            if maze[i][j] == Tile.START:
                start = (i, j)
            elif maze[i][j] == Tile.END:
                end = (i, j)
    
    assert start and end, "No start or end"

    queue = deque([start])
    visited.add(start)

    # BFS
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break  # Found the end

        possible_paths = 0
        for dx, dy in move_directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited and maze[nx][ny] != Tile.WALL:
                queue.append((nx, ny))
                visited.add((nx, ny))
                prev[nx][ny] = (x, y)
                distance[nx][ny] = distance[x][y] + 1
                possible_paths += 1
        
        if possible_paths > 1:
            decision_point.add((x, y))

    # Reconstruct path
    path = []
    at = end
    while at:
        path.append(at)
        at = prev[at[0]][at[1]]
    path.reverse()

    def get_direction(dx, dy):
        if dx == 1:
            return Tile.SOUTH
        elif dx == -1:
            return Tile.NORTH
        elif dy == 1:
            return Tile.EAST
        elif dy == -1:
            return Tile.WEST

    # Convert path to directions
    directions = []
    markers = []
    
    def add_marker(marker, relative_pos=0, marker_metadata=None):
        markers.append((marker.value, len(directions) + relative_pos, marker_metadata))
    
    def is_move_possible(x, y):
        return 0 <= x < n and 0 <= y < n and maze[x][y] != Tile.WALL

    def add_direction(direction):
        directions.append(direction.value)

        if direction == Tile.SOUTH:
            add_marker(Markers.FALLIBLE_GOES_SOUTH)
        if direction == Tile.NORTH:
            add_marker(Markers.FALLIBLE_GOES_NORTH)
        if direction == Tile.EAST:
            add_marker(Markers.FALLIBLE_GOES_EAST)
        if direction == Tile.WEST:
            add_marker(Markers.FALLIBLE_GOES_WEST)
        



    for i in range(1, len(path)):
        x, y = path[i]

        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]

        add_direction(get_direction(dx, dy))

        if is_move_possible(x+1, y):
            add_marker(Markers.SOUTH_IS_POSSIBLE)
        if is_move_possible(x-1, y):
            add_marker(Markers.NORTH_IS_POSSIBLE)
        if is_move_possible(x, y+1):
            add_marker(Markers.EAST_IS_POSSIBLE)
        if is_move_possible(x, y-1):
            add_marker(Markers.WEST_IS_POSSIBLE)


        if (x, y) in decision_point:
            add_marker(Markers.DECISION)
                
        if (x, y) in decision_point:
            if random.random() < MISTAKE_PROBABILITY:
                other_possible_paths = []

                x2, y2 = path[i+1]

                dx2 = x2 - x
                dy2 = y2 - y
                correct_direction = get_direction(dx2, dy2)

                for dx, dy in move_directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] != Tile.WALL and (nx, ny) not in path:
                        other_possible_paths.append((nx, ny))

                new_choice_x, new_choice_y = random.choice(other_possible_paths)
                dx = new_choice_x - x
                dy = new_choice_y - y

                mistake_move = get_direction(dx, dy)
                mistake_correction_move = get_direction(-dx, -dy)

                add_direction(mistake_move)
                add_marker(Markers.MISTAKE, 0, correct_direction)
                add_direction(mistake_correction_move)
                add_marker(Markers.CORRECTION)
                # todo: maybe this is also a decision again?
            else:
                # todo: maybe it sees this as a deicison again?
                add_marker(Markers.NON_MISTAKE, 1)

    

    return directions, markers


maze = generate_maze()
directions, decision_steps = find_shortest_path_with_directions(maze)
print(serialize_sample(maze, directions))
display_maze_with_markers(maze, directions, decision_steps)


# %%
import json
import tqdm
# Assume your functions are defined here

def main():
    quantity = 50_000

    file_path = os.path.join(os.path.dirname(__file__), "correctable/data.jsonl")
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'a') as file:
        for _ in tqdm.tqdm(range(quantity)):
            maze = generate_maze()
            directions, markers = find_shortest_path_with_directions(maze)
            # Convert the maze to a JSON-compatible format
            row = {
                "maze": "\n".join(["".join([tile.value for tile in row]) for row in maze]),
                "directions": "".join(directions),
                "markers": markers
            }

            # Use string interpolation to create the JSON string manually
            record_json = json.dumps(row)

            # Write the JSON string to a file, appending a newline to form the JSON Lines format
            file.write(record_json + '\n')

main()

# %%
