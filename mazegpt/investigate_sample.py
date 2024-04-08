

# %%
from gytis import imports


# %%

from mazegpt.utils import display_maze, parse_maze
output = """###############
#s  #         #
### # ####### #
# # #   # #   #
# # ### # # ###
# #     # # # #
# ####### # # #
#     #   #   #
# ### # # ### #
# #   # # #   #
# ##### # # ###
#       #   # #
# ########### #
#            e#
###############;EESSSSEEEENNWWNNEEEEEESSWWSSSSEESSWWNNNNWWSSSSWWWWSSEEEEEEEE;
"""

maze, directions = parse_maze(output)
display_maze(maze, directions)

# %%

# %%
