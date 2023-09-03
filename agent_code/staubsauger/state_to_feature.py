import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V14(game_state)

def state_to_features_V9(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V8, but added safe tiles outside of the explosio radius
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    bombs = game_state["bombs"]

	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, player)

    # Determin distance to nearest coin
	# Add area_around_player to feature_vector
    feature_vector += find_min_coin_relative_coordinate(coins, player)

    # Avoid bomb

    # There is no danger
    best_safe_tile = [-1,-1] 

    if within_explosion_radius(*get_player_coordinates(player), field, bombs):
        playerX, playerY = get_player_coordinates(player)

        dangerous_bomb = find_closest_dangerous_bomb(bombs, field, (playerX, playerY))
        blast_coords = get_blast_coords(dangerous_bomb[0], field)
        reachable_tiles = get_reachable_tiles((playerX, playerY), field)
        safe_tiles = get_safe_tiles(reachable_tiles, blast_coords)

        # Add tiles where you are safe from dangerous bomb to feature vector
        feature_vector += tuple(safe_tiles)

        # add distance to dangerous bomb
        dist_to_closest_bomb = np.linalg.norm(np.array(playerX, playerY) - np.array((dangerous_bomb[0][0], dangerous_bomb[0][1])))
        feature_vector += tuple([dist_to_closest_bomb])

	# Return hash value of area around player
    key = hash(feature_vector)
    return key

def state_to_features_V10(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V9, but no area around player - this fixed the running in a loop and standing around stupidly
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]

	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    #feature_vector = get_area_around_player(field, player)

    feature_vector += tuple(field.flatten())
    # Determin distance to nearest coin
	# Add area_around_player to feature_vector
    feature_vector += find_min_coin_relative_coordinate(coins, player)

    # Avoid bomb
    playerX, playerY = get_player_coordinates(player)
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, (playerX, playerY))
    # add distance to dangerous bomb
    if dangerous_bomb == None:
        dist_to_closest_bomb = 0
    else:
        dist_to_closest_bomb = np.linalg.norm(np.array(playerX, playerY) - np.array((dangerous_bomb[0][0], dangerous_bomb[0][1])))

    feature_vector += tuple([dist_to_closest_bomb])
    feature_vector += tuple(explosions.flatten())

	# Return hash value of area around player
    key = hash(feature_vector)
    return key

def state_to_features_V11(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V9, but only reachable tiles - makes the agent do literally nothing
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]

	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
    reachable_tiles = get_reachable_tiles(get_player_coordinates(player), field)

    feature_vector += tuple(reachable_tiles)

    # Determin distance to nearest coin
	# Add area_around_player to feature_vector
    feature_vector += find_min_coin_relative_coordinate(coins, player)

    # Avoid bomb
    playerX, playerY = get_player_coordinates(player)
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, (playerX, playerY))
    # add distance to dangerous bomb
    if dangerous_bomb == None:
        dist_to_closest_bomb = 0
    else:
        dist_to_closest_bomb = np.linalg.norm(np.array(playerX, playerY) - np.array((dangerous_bomb[0][0], dangerous_bomb[0][1])))

    feature_vector += tuple([dist_to_closest_bomb])
    feature_vector += tuple(explosions.flatten())

	# Return hash value of area around player
    key = hash(feature_vector)
    return key

def state_to_features_V12(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Same as V11, but given the safe tiles
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]

	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
    if within_explosion_radius(*get_player_coordinates(player), field, bombs):
        playerX, playerY = get_player_coordinates(player)

        reachable_tiles = get_reachable_tiles((playerX, playerY), field)
        safe_tiles = get_safe_tiles(reachable_tiles, bombs, field)

        # Add tiles where you are safe from dangerous bomb to feature vector
        feature_vector += tuple(safe_tiles)

    # Determin distance to nearest coin
	# Add area_around_player to feature_vector
    feature_vector += find_min_coin_relative_coordinate(coins, player)

    # Avoid bomb
    playerX, playerY = get_player_coordinates(player)
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, (playerX, playerY))
    # add distance to dangerous bomb
    if dangerous_bomb == None:
        dist_to_closest_bomb = 0
    else:
        dist_to_closest_bomb = np.linalg.norm(np.array(playerX, playerY) - np.array((dangerous_bomb[0][0], dangerous_bomb[0][1])))

    feature_vector += tuple([dist_to_closest_bomb])
    feature_vector += tuple(explosions.flatten())

	# Return hash value of area around player
    key = hash(feature_vector)
    return key

def state_to_features_V13(game_state: dict) -> int:
    """
    Converts the game state into a number
    Try to calcualte the distance to the nearest bomb and maximize it
    Note: 
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    coins = game_state["coins"]
    player = game_state["self"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]

	# Stores all the relevant information that is passed on to the agent
    feature_vector = ()
    
	# Add area_around_player to feature_vector
    feature_vector = get_area_around_player(field, player)
    
	# Determin distance to nearest coin
	# Add area_around_player to feature_vector
    feature_vector += find_min_coin_relative_coordinate(coins, player)    

    # Avoid bomb
    feature_vector += get_min_bomb_relative_coordinate(bombs, field, player)
    
	# Check for explosions around you
    feature_vector += check_for_explosions_around(explosions, player)

	# Return hash value of area around player
    key = hash(feature_vector)
    return key


def state_to_features_V14(game_state: dict) -> int:
    """
    Converts the game state into a number
    :param game_state:  A dictionary describing the current game board.
    Note: Way to many states
    :return: int
    """
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