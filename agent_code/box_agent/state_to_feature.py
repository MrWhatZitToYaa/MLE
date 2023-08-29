import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V12(game_state)

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
    feature_vector += find_min_coin_coordinate(coins, player)

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
    feature_vector += find_min_coin_coordinate(coins, player)

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
    feature_vector += find_min_coin_coordinate(coins, player)

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
    feature_vector += find_min_coin_coordinate(coins, player)

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