

# %%
from gytis import imports


# %%

from mazegpt.utils import display_maze, parse_maze
# output = """s  #         #
# ## # ##### # #
#    # #     # #
#  ##### ##### #
#    #   #     #
# ## # ### #####
#  # #   # #   #
#  # # # # # # #
#    # # # # # #
#  ### # ### # #
#    # #     # #
# ## ######### #
#             e#
# ##############;EESSWWSSEESSSSWWN;
# """

output="""s#   
 ### 
     
#### 
    e;SSEEEENNEESSSS;
"""

# output = """s  #         #
# ## # ##### # #
#    # #     # #
#  ##### ##### #
#    #   #     #
# ## # ### #####
#  # #   # #   #
#  # # # # # # #
#    # # # # # #
#  ### # ### # #
#    # #     # #
# ## ######### #
#             e#
# ##############;EESSWWSSEESSSSWWSSEESSEEEEEEEE;
# """
maze, directions = parse_maze(output)
display_maze(maze, directions)

# %%

# %%
