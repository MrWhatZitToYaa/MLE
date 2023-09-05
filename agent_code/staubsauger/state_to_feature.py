import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V15(game_state)

def state_to_features_V8(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V7, but use rel distance instead of direction
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"].flatten()
    coins = game_state["coins"]
    player = game_state["self"]
    
	# Transform arena
    field = [list_of_blocks.EMPTY.value if x == 0 else 
             list_of_blocks.BRICK.value if x == -1 else
             list_of_blocks.CRATE.value
             for x in field]
    
	# Player cords
    playerX = player[3][0]
    playerY = player[3][1]

	# Init area, only use odd sizes
    area_size = 3
    area_around_player = np.zeros((area_size, area_size))
    
    #Copy area around player
    for i in range(0,area_size):
         for j in range(0,area_size):
              area_around_player[i][j] = field[(i-1+playerY)*ARENA_LENGTH + j-1 + playerX]
         
	# Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value
    
    area_around_player = tuple(area_around_player.flatten())
    
	# Add (TODO: try relative) coordinates of nearest coin(s)
    nearestCoin = [-1,-1]
    nearestCoinXdist = ARENA_LENGTH + 1
    nearestCoinYdist = ARENA_WIDTH  + 1
    
    for i in coins:
         coinX = i[0]
         coinY = i[1]
         if abs(coinX - playerX) + abs(coinY - playerY) < nearestCoinXdist + nearestCoinYdist:
                nearestCoinXdist = abs(coinX - playerX)
                nearestCoinYdist = abs(coinY - playerY)
                nearestCoin = i

    # Relative directino of coin to player
    directionX = nearestCoin[0] - playerX
    directionY = nearestCoin[1] - playerY
    
    area_around_player += tuple([directionY, directionX])
	# Return hash value of area around player
    key = hash(area_around_player)
    return key

def state_to_features_V15(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V7, but use rel distance instead of direction
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    
	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, player)
    
	# Add direction to nearest coin to feature_vector
    min_dist_coin_X, min_dist_coin_Y = find_min_coin_relative_coordinate(coins, player)
    feature_vector += get_direction_for_coin(min_dist_coin_X, min_dist_coin_Y)
    
	# Return hash value of feature_vector
    key = hash(feature_vector)
    return key

def state_to_features_V16(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V7, but use rel distance instead of direction
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    bombs = game_state["bombs"]
    coins = game_state["coins"]
    player = game_state["self"]
    
	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, player)
    
	# Add direction to nearest coin to feature_vector
    min_dist_coin_X, min_dist_coin_Y = find_min_coin_relative_coordinate(coins, player)
    feature_vector += get_direction_for_coin(min_dist_coin_X, min_dist_coin_Y)
    
	# Add direction to run away from bomb
    player_coords = get_player_coordinates(player)    
    for bomb in bombs:
        radius = get_blast_coords(bomb[0], field)

	# Return hash value of feature_vector
    key = hash(feature_vector)
    return key