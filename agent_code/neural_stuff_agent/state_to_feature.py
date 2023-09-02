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
    modified_field = np.append(modified_field, [player[2]])
    
    return tuple(modified_field)