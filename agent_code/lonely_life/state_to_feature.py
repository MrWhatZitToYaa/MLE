import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V18(game_state)

# Multiplayer approved
def state_to_features_V18(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V7, but use rel distance instead of direction
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    enemies = game_state["others"]
    
	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, explosions, player, bombs, enemies)
    
	# Add direction to nearest coin to feature_vector
    feature_vector += get_direction_for_coin(coins, player, field)
    
		# Debugging prupese
    if(game_state["step"] == 3):
         print("hi")

	# Add direction to run away from bomb
    feature_vector += get_direction_for_safe_tile(bombs, player, field)
    
	# Add direction for nearest crate
    feature_vector += get_direction_for_crate(player, field)
    
	# Add if bomb can be dropped without dying
    feature_vector += get_safe_bomb_drop(player, field)
    
	# Add direction and distance to nearest enemy
    # feature_vector += get_ememy_information(player, enemies, field)



	# Return hash value of feature_vector
    key = hash(feature_vector)
    return key