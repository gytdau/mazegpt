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

from mazegpt.utils import display_maze, parse_maze

maze, directions = parse_maze(output)

# %%
display_maze(maze, directions)

# %%
