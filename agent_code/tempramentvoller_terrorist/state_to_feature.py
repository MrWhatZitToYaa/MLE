import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V17(game_state)

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
def state_to_features_V14(game_state: dict) -> int:
    """
    Converts the game state into a number
    :param game_state:  A dictionary describing the current game board.
    Note: Way to many states
    :return: int
    """
    # Takes very long to train tbh
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    enemies = game_state["others"]

    # Stores all the relevant information that is passed on to the agent
    modified_field = np.zeros((ARENA_LENGTH, ARENA_WIDTH))

    # Transform arena
    field = field.flatten()
    transformed_field = [list_of_blocks.EMPTY.value if x == 0 else
                         list_of_blocks.BRICK.value if x == -1 else
                         list_of_blocks.CRATE.value
                         for x in field]

    transformed_field = np.reshape(transformed_field, (ARENA_LENGTH, ARENA_WIDTH))

    modified_field = transformed_field

    # Add bombs to field
    for bomb in bombs:
        tick = bomb[1]
        bombX = bomb[0][0]
        bombY = bomb[0][1]
        block = list_of_blocks.BOMB_TICK0.value

        if tick == 3:
            block = list_of_blocks.BOMB_TICK3.value
        if tick == 2:
            block = list_of_blocks.BOMB_TICK2.value
        if tick == 1:
            block = list_of_blocks.BOMB_TICK1.value

        modified_field[bombX][bombY] = block

    # Add explosions to field
    for i in range(ARENA_LENGTH):
        for j in range(ARENA_WIDTH):
            explosion = explosions[i][j]

            if explosion == 2:
                block = list_of_blocks.EXPLOSION2.value
                modified_field[i][j] = block
            if explosion == 1:
                block = list_of_blocks.EXPLOSION1.value
                modified_field[i][j] = block

    # Add coins to field
    for coin in coins:
        coinX = coin[0]
        coinY = coin[1]

        modified_field[coinX][coinY] = list_of_blocks.COIN.value

    # Add player to field
    playerX = player[3][0]
    playerY = player[3][1]

    modified_field[playerX][playerY] = list_of_blocks.PLAYER.value

    # Add enemies to field
    for enemy in enemies:
        enemyX = enemy[3][0]
        enemyY = enemy[3][1]

        modified_field[enemyX][enemyY] = list_of_blocks.ENEMY.value

    modified_field = modified_field.flatten()

    # Can throw bomb
    np.append(modified_field, [player[2]])

    return tuple(modified_field)

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
    feature_vector += get_directions_for_object(min_dist_coin_X, min_dist_coin_Y)
    
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
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    
	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, explosions, player)
    
	# Add direction to nearest coin to feature_vector
    min_dist_coin_X, min_dist_coin_Y = find_min_coin_relative_coordinate(coins, player)
    feature_vector += get_directions_for_object(min_dist_coin_X, min_dist_coin_Y)
    
	# Add direction to run away from bomb
    direction_to_closest_safe_tile_X, direction_to_closest_safe_tile_Y = -2, -2
    if(within_explosion_radius(player, field, bombs)):
         reachable_tiles = get_reachable_tiles(get_player_coordinates(player), field, 3)
         safe_tiles = get_safe_tiles(reachable_tiles, bombs, field)
         if(len(safe_tiles) == 0):
              direction_to_closest_safe_tile_X, direction_to_closest_safe_tile_Y = -3, -3
         else:
              direction_to_closest_safe_tile_X, direction_to_closest_safe_tile_Y = get_direction_to_closetes_safe_tile(player, safe_tiles)
         
    feature_vector += (direction_to_closest_safe_tile_X, direction_to_closest_safe_tile_Y)
    
	# Add direction to nearest crate
    min_dist_crate_X, min_dist_crate_Y = find_min_crate_relative_coordinate(field, player)
    feature_vector += get_directions_for_object(min_dist_crate_X, min_dist_crate_Y)

	# Return hash value of feature_vector
    key = hash(feature_vector)
    return key

def state_to_features_V17(game_state: dict) -> int:
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
    
	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, explosions, player)
    
	# Add direction to nearest coin to feature_vector
    feature_vector += get_direction_for_coin(coins, player, field)
    
	# Add direction to run away from bomb
    feature_vector += get_direction_for_safe_tile(bombs, player, field)
    
	# Add direction for nearest crate
    feature_vector += get_direction_for_crate(player, field)
    
	# Add if bomb can be dropped
    feature_vector += (player[2],)

	# Return hash value of feature_vector
    key = hash(feature_vector)
    return key