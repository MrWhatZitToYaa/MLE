import numpy as np

from .definitions import *

def state_to_features(game_state: dict) -> int:
     """
     Use this function to choose the appropiate state to feature method implementation
     :param game_state:  A dictionary describing the current game board.
     :return: int
     """
     return state_to_features_V8(game_state)

def state_to_features_V1(game_state: dict) -> int:
    """
    Converts the game state into a number
    :param game_state:  A dictionary describing the current game board.
    Note: Way to many states
    :return: int
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    round = game_state["round"]
    step = game_state["step"]
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    enemies = game_state["others"]
    
    round = tuple([round])
    step = tuple([step])
    field = tuple(field.flatten())
    bombs = tuple(np.array([np.array([i[0][0], i[0][1], i[1]]) for i in bombs]).flatten())
    explosions = tuple(explosions.flatten())
    coins = tuple(np.array([np.array([i[0], i[1]]) for i in coins]).flatten())
    player = (player[0], player[1], player[2], player[3][0],player[3][1])
    enemies = [np.array([i[0], i[1], i[2], i[3][0],i[3][1]]) for i in enemies]
    enemies_flat = []
    for sublist in enemies:
         for item in sublist:
              enemies_flat.append(item)
    enemies_flat = tuple(enemies_flat)
              
    total_list = round + step + field + bombs + explosions + coins + player + enemies_flat 
    
    return total_list

def state_to_features_V2(game_state: dict) -> int:
    """
    Converts the game state into a number
    Note: Still way to many states
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    field = game_state["field"]
    player = game_state["self"]
    
    field = tuple(field.flatten())
    player = (player[3][0],player[3][1])
              
    total_list = field + player 
    
    return total_list

def state_to_features_V3(game_state: dict) -> int:
    """
    Converts the game state into a number
    Note: Simular but better than 2, but still way to many states
    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    
    field = game_state["field"]
    field = field.flatten()
    coins = game_state["coins"]
    player = game_state["self"]

	# Init arena
    arena = np.zeros((ARENA_LENGTH, ARENA_WIDTH))
    
	# Transform arena
    field = [list_of_blocks.EMPTY.value if x == 0 else 
             list_of_blocks.BRICK.value if x == -1 else
             list_of_blocks.CRATE.value
             for x in field]
    
    for i in range(0,ARENA_LENGTH):
         for j in range(0,ARENA_WIDTH):
              arena[i][j] = field[i*ARENA_LENGTH + j]
    
	# Add coins to arena
    for i in coins:
         arena[i[1]][i[0]] = list_of_blocks.COIN.value
         
	# Add player to arena
    arena[player[3][1]][player[3][0]] = list_of_blocks.PLAYER.value
    
    arena = tuple(arena.flatten())
    
    key = hash(arena)
    return key

def state_to_features_V4(game_state: dict) -> int:
    """
    Converts the game state into a number
    Note: Works well but gets stuck if no coins are found in surrounding area
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
              
	# Add coins around player to area
    for i in coins:
         coinX = i[0]
         coinY = i[1]
         if abs(coinX - playerX) <= 1 and abs(coinY - playerY) <= 1:
              coinX = coinX - playerX + 1
              coinY = coinY - playerY + 1
              area_around_player[coinY][coinX] = list_of_blocks.COIN.value
         
	# Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value
    
	# Return hash value of area around player
    area_around_player = tuple(area_around_player.flatten())
    key = hash(area_around_player)
    return key

def state_to_features_V5(game_state: dict) -> int:
    """
    Converts the game state into a number
    Same as V4 but adds location of nearest coin(s)
    Note: Results in too many states and bot doesn't see every state so gets lost
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
              
	# Add coins around player to area
    for i in coins:
         coinX = i[0]
         coinY = i[1]
         if abs(coinX - playerX) <= 1 and abs(coinY - playerY) <= 1:
              coinX = coinX - playerX + 1
              coinY = coinY - playerY + 1
              area_around_player[coinY][coinX] = list_of_blocks.COIN.value
         
	# Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value
    
    area_around_player = tuple(area_around_player.flatten())
    
	# Add (TODO: try relative) coordinates of nearest coin(s)
    nearestCoin = [0,0]
    nearestCoinXdist = ARENA_LENGTH + 1
    nearestCoinYdist = ARENA_WIDTH  + 1
    
    for i in coins:
         coinX = i[0]
         coinY = i[1]
         if abs(coinX - playerX) + abs(coinY - playerY) < nearestCoinXdist + nearestCoinYdist:
                nearestCoinXdist = abs(coinX - playerX)
                nearestCoinYdist = abs(coinY - playerY)
                nearestCoin = i
                
    area_around_player = area_around_player + tuple([nearestCoin[1], nearestCoin[0]])
    
	# Return hash value of area around player
    key = hash(area_around_player)
    return key

def state_to_features_V6(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
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
    nearestCoin = [0,0]
    nearestCoinXdist = ARENA_LENGTH + 1
    nearestCoinYdist = ARENA_WIDTH  + 1
    
    for i in coins:
         coinX = i[0]
         coinY = i[1]
         if abs(coinX - playerX) + abs(coinY - playerY) < nearestCoinXdist + nearestCoinYdist:
                nearestCoinXdist = abs(coinX - playerX)
                nearestCoinYdist = abs(coinY - playerY)
                nearestCoin = i
                
    area_around_player += tuple([nearestCoin[1], nearestCoin[0]])
    area_around_player += tuple([playerY, playerX])
    
	# Return hash value of area around player
    key = hash(area_around_player)
    return key

def state_to_features_V7(game_state: dict) -> int:
    """
    Converts the game state into a number
    Only player location, walls arround him and nearest coin
    Note: Words well but gets stuck too
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
    directionX = -1
    directionY = -1
    if nearestCoin[0] < playerX:
        directionX = 0
    elif nearestCoin[0] == playerX:
        directionX = 1
    else:
        directionX = 2

    if nearestCoin[1] < playerY:
        directionY = 0
    elif nearestCoin[1] == playerY:
        directionY = 1
    else:
        directionY = 2
    area_around_player += tuple([directionY, directionX])
	# Return hash value of area around player
    key = hash(area_around_player)
    return key

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

def find_min_distance(coins, playerX, playerY):
    min_d = ARENA_LENGTH
    for (coinX, coinY) in coins:
        d = np.linalg.norm(np.array((playerX, playerY)) - np.array((coinX, coinY)))
        if d < min_d:
            min_d = d
    return min_d
