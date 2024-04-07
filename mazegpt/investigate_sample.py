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
from gytis import imports


# %%

from mazegpt.utils import create_maze_html, parse_maze

maze, directions = parse_maze(output)

# %%
create_maze_html(maze, directions)

# %%
