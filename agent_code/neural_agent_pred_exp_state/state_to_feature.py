import numpy as np

from .definitions import *
from .state_to_feature_helpers import *

def state_to_features(game_state: dict):
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     # TODO: train with sarsa features alone on different scenarios and plot
     # TODO: train with NN features on different scenarios and plot
     return state_to_features_NN(game_state)
     #return state_to_features_SARSA(game_state)

def state_to_features_SARSA(game_state: dict) -> int:
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

def state_to_features_NN(game_state: dict):
    if game_state is None:
        return None

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    enemies = game_state["others"]

    # Stores all the relevant information that is passed on to the agent
    feature_vec = np.zeros((NUM_FEATURES,ARENA_LENGTH, ARENA_WIDTH))

    transformed_field = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
    # Transform arena
    field = field.flatten()
    transformed_field = [list_of_blocks.EMPTY.value if x == 0 else
                         list_of_blocks.BRICK.value if x == -1 else
                         list_of_blocks.CRATE.value
                         for x in field]

    transformed_field = np.reshape(transformed_field, (ARENA_LENGTH, ARENA_WIDTH))

    feature_vec[0:,:] = transformed_field

    bomb_map = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
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

        bomb_map[bombX][bombY] = block

    feature_vec[1,:,:,] = bomb_map

    # Add explosions to field
    explosion_map = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
    for i in range(ARENA_LENGTH):
        for j in range(ARENA_WIDTH):
            explosion = explosions[i][j]

            if explosion == 2:
                block = list_of_blocks.EXPLOSION2.value
                explosion_map[i][j] = block
            if explosion == 1:
                block = list_of_blocks.EXPLOSION1.value
                explosion_map[i][j] = block

    feature_vec[2,:,:,] = explosion_map

    coin_map = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
    # Add coins to field
    for coin in coins:
        coinX = coin[0]
        coinY = coin[1]

        coin_map[coinX][coinY] = list_of_blocks.COIN.value

    feature_vec[3,:,:,] = coin_map

    players_map = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
    # Add player to field
    playerX = player[3][0]
    playerY = player[3][1]

    players_map[playerX][playerY] = list_of_blocks.PLAYER.value

    # Add enemies to field
    for enemy in enemies:
        enemyX = enemy[3][0]
        enemyY = enemy[3][1]

        players_map[enemyX][enemyY] = list_of_blocks.ENEMY.value

    feature_vec[4,:,:] = players_map
    #print('shape of feature vec:', feature_vec.shape)
    #print('field_map:', transformed_field)
    #print('bomb_map:', bomb_map)
    #print('coin_map:', coin_map)
    #print('players_map:', players_map)
    #print('explosion_map:', explosion_map)

    #print('features:',feature_vec.shape)
    feature_vec_flat = feature_vec.flatten()
    #print(feature_vec_flat)
    return feature_vec_flat
