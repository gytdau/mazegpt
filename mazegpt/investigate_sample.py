

# %%
from gytis import imports
from mazegpt.sample import model

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

output="""s  #     #       # 
## # ### # ### # # 
 # # # # #   # # # 
 # # # # ### # # # 
 #   #   #   # #   
 ##### ####### ####
 #     #     #     
 # ####### # # ### 
 #       # # # #   
 ####### # # ### # 
   #   #   # #   # 
 # ### ##### # ### 
 #         #   # # 
 ############### # 
       #         # 
 ##### # ##### # # 
 #   # # # #   # # 
 ### # # # # ##### 
     #     #      e;EESSSSEENNNNEEEESSSSWWNSSSWWWWSSEEEEEESSEENNNNEESSSSSSEENNEESSSSSS;
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
