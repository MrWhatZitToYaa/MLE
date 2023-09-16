import numpy as np
from scipy.spatial.distance import cdist
from collections import deque

from .definitions import *

def get_player_coordinates(player):
    """
    returns player coordiantes
    :params player "self" in the state space
    """
    playerX = player[3][0]
    playerY = player[3][1]
    
    return (playerX, playerY)

def get_area_around_player(field, explosions, player, bombs):
    """
    returns the 3x3 area arround the player, including deadly tiles and explosions
    """
    field = field.flatten()
    # Transform arena
    transformed_field = [list_of_blocks.EMPTY.value if x == 0 else 
                        list_of_blocks.BRICK.value if x == -1 else
                        list_of_blocks.CRATE.value
                        for x in field]
    
    transformed_field = np.array(transformed_field).reshape(ARENA_LENGTH, ARENA_WIDTH)
    
    playerX, playerY = get_player_coordinates(player)

	# Init area, only use odd sizes
    area_around_player_size = 3
    area_around_player = np.zeros((area_around_player_size, area_around_player_size))
    
    #Copy area around player
    for i in range(0,area_around_player_size):
         for j in range(0,area_around_player_size):
              area_around_player[i][j] = transformed_field[i-1 + playerX][j-1 + playerY]
         
	# Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value
    
	# Add explosions to area
    sizeX, sizeY = explosions.shape
    for x in range(sizeX):
        for y in range(sizeY):
            if(explosions[x][y] != 0):
                if(abs(playerX - x) + abs(playerY - y) <= 1):
                    explosion_rel_X = x - playerX
                    explosion_rel_Y = y - playerY
                    area_around_player[explosion_rel_X+1][explosion_rel_Y+1] = list_of_blocks.EXPLOSION0.value
                    
	# Add lethal tiles to the area
	# Optional propoal: only add lethal tiles when no safe tile, because, when safetile exists agaent
	# knows danger
    field = field.reshape(ARENA_LENGTH, ARENA_WIDTH)
    
    reachable_tiles = []
    for dx, dy in [(-1,-1), (0,-1), (1,-1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]:
            x = playerX + dx
            y = playerY + dy
            if(field[x][y] == 0 and (playerX != x and playerY != y)):
                reachable_tiles.append((x,y))
    

    for bomb in bombs:
        for tile in reachable_tiles:
            if tile in get_blast_coords(bomb[0], field):
                soon_danger_X = tile[0] - playerX
                soon_danger_Y = tile[1] - playerY
                area_around_player[soon_danger_X+1][soon_danger_Y+1] = list_of_blocks.DANGER.value
                    

	# Add area_around_player to feature_vector
    return tuple(area_around_player.flatten())

def get_safe_tiles(reachable_tiles, bombs, field):
    """
    return all tiles not in the blast radius of the bombs
    """
    safe_tiles = []
    if bombs == []: return reachable_tiles
    else:
        for bomb in bombs:
            for tile in reachable_tiles:
                if tile not in get_blast_coords(bomb[0], field):
                    safe_tiles.append(tile)
    return safe_tiles

def find_min_coin_coordinate(coins: list, player):
    """
    returns the coordiantes of the closest coin
    """
    coords = find_min_object_coordiantes(coins, player)   
    return coords

def find_min_safe_tile_coordinate(safe_tiles, player):
    """
    returns the coordiantes of the closest safe tile
    """
    coords = find_min_object_coordiantes(safe_tiles, player)   
    return coords

def find_min_crate_coordinate(field, player):
    """
    returns the coordiantes of the closest crate
    """
	# Get all the crates from the field
    crates = []
    for x in range(ARENA_WIDTH):
        for y in range(ARENA_LENGTH):
            if field[x][y] == 1:
                crates.append((x,y))

    coords = find_min_object_coordiantes(crates, player)   
    return coords

def find_min_object_coordiantes(obj: list, player):
    """
    returns the coordiantes of the closest object
    """
    playerX, playerY = get_player_coordinates(player)

    x = ARENA_WIDTH
    y = ARENA_LENGTH
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (objX, objY) in obj:
        d = abs(objX - playerX) + abs(objY - playerY)
        if d < min_d:
            min_d = d
            x = objX
            y = objY
    return (x, y)

def find_min_coin_distance(coins: list, playerX, playerY):
    """
    returns the distance of the closest coin
    """
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (coinX, coinY) in coins:
        d = abs(coinX - playerX) + abs(coinY - playerY)
        if d < min_d:
            min_d = d
    return min_d

def get_direction_for_safe_tile(bombs, player, field):
    """
    returns the direction of the closest safe tile
    TODO: CHECK
    """
    safe_tiles = []
    if(within_explosion_radius(player, field, bombs)):
         reachable_tiles = get_reachable_tiles(get_player_coordinates(player), field, 3)
         safe_tiles = get_safe_tiles(reachable_tiles, bombs, field)
    
	# No safe tile
    if(len(safe_tiles) == 0):
        return (list_of_steps.NO_TARGET.value,)
    # Get closest safe tile
    safe_tile_X, safe_tile_Y = find_min_safe_tile_coordinate(safe_tiles, player)

    direction = get_direction_for_object(safe_tile_X, safe_tile_Y, player, field)
    if direction == None:
        return (list_of_steps.NODIR.value,)
    return direction

def get_direction_for_coin(coins, player, field):
    coin_X, coin_Y = find_min_coin_coordinate(coins, player)
    
	# None found or on top
    playerX, playerY = get_player_coordinates(player)
    if(coin_X == ARENA_LENGTH and coin_Y == ARENA_WIDTH) or (coin_X == playerX and coin_Y == playerY):
        return (list_of_steps.NODIR.value,)

    direction = get_direction_for_object(coin_X, coin_Y, player, field)
    if direction == None:
        return (list_of_steps.NODIR.value,)
    return direction

def get_direction_for_crate(player, field):
    crate_X, crate_Y = find_min_crate_coordinate(field, player)
    
	# None found or on top
    playerX, playerY = get_player_coordinates(player)
    if(crate_X == ARENA_LENGTH and crate_Y == ARENA_WIDTH) or (crate_X == playerX and crate_Y == playerY):
        return (list_of_steps.NODIR.value,)

    direction = get_direction_for_object(crate_X, crate_Y, player, field)
    if direction == None:
        return (list_of_steps.NODIR.value,)
    return direction

def get_direction_for_object(objX, objY, player, field):
    path = find_path(field, get_player_coordinates(player), (objX, objY))
    direction = None
    if path is not None:
        if len(path) < 2:
            firstStepX, firstStepY = path[0]
        else:
            firstStepX, firstStepY = path[1]
        playerX, playerY = get_player_coordinates(player)

        direction = (-1)
        # Find the relative direction to the player
        if(firstStepX - playerX > 0):
            direction = list_of_steps.RIGHT.value
        if(firstStepX - playerX < 0):
            direction = list_of_steps.LEFT.value
        if(firstStepY - playerY > 0):
            direction = list_of_steps.DOWN.value
        if(firstStepY - playerY < 0):
            direction = list_of_steps.UP.value
        
    return (direction,)

def find_path(field, start, end):
    def is_valid(x, y):
        return 0 <= x < ARENA_WIDTH and 0 <= y < ARENA_LENGTH and (field[x][y] == 0 or field[x][y] == field[end[0]][end[1]]) and not visited[x][y]

    visited = [[False for _ in range(17)] for _ in range(17)]
    parent = [[None for _ in range(17)] for _ in range(17)]

    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        x, y = queue.popleft()

        if (x, y) == end:
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[x][y]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y):
                visited[new_x][new_y] = True
                parent[new_x][new_y] = (x, y)
                queue.append((new_x, new_y))

    return None

def find_min_bomb_relative_coordinate(bomb: tuple, player):
    playerX, playerY = get_player_coordinates(player)
    bombX, bombY = bomb[0]

    min_x = bombX - playerX
    min_y = bombY - playerY
    
    return (min_x, min_y)

def get_min_bomb_relative_coordinate(bombs, field, player):
    playerX, playerY = get_player_coordinates(player)
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, (playerX, playerY))
    
	# Add distance to dangerous bomb
    bombDistX = ARENA_WIDTH
    bombDistY = ARENA_LENGTH
	
    if dangerous_bomb == None:
        pass
    else:
        bombDistX, bombDistY = find_min_bomb_relative_coordinate(dangerous_bomb, player)
        
    return (bombDistX, bombDistY)

def get_blast_coords(bomb_coords, field):
    x, y = bomb_coords[0], bomb_coords[1]
    blast_coords = [(x, y)]

    for i in range(1, 3):
        if field[x + i][y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, 3):
        if field[x - i][y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, 3):
        if field[x][y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, 3):
        if field[x][y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords

def within_explosion_radius(player, field, bombs):
    """
    Checks whether the agent is within the radius of a bomb. It doesn't take into account when the bomb is set to go off.
    """
    player_coords = get_player_coordinates(player)
    for bomb in bombs:
        radius = get_blast_coords(bomb[0], field)
        if player_coords in radius:
            return True
    return False

def find_closest_dangerous_bomb(bombs, field, player_coords):
    """
    This finds a bomb wihtin whose radius the agent is. IMPORTANT: There may be more than one dangerous bomb, this will only find the closest.
    """
    dangerous_bombs = []
    for bomb in bombs:
        radius = get_blast_coords(bomb[0], field)
        if player_coords in radius:
            dangerous_bombs.append(bomb)

    closest_dangerous_bomb = None
    min_d = ARENA_LENGTH
    for bomb in dangerous_bombs:
        #d = np.linalg.norm(np.array((player_coords[0], player_coords[1])) - np.array((bomb[0][0], bomb[0][1])))
        # Manhattan dist
        player_arr = np.array((player_coords[0], player_coords[1]))
        bomb_arr = np.array((bomb[0][0], bomb[0][1]))
        d = sum(abs(player_arr - coin_arr) for player_arr, coin_arr in zip(player_arr, bomb_arr))
        if d < min_d:
            min_d = d
            closest_dangerous_bomb = bomb
    return closest_dangerous_bomb


def check_for_explosions_around(explosions, player):
    playerX, playerY = get_player_coordinates(player)
    
    explosion_near_player = False
    
    num_rows, num_cols = explosions.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if((abs(i - playerY) <= 1) and (abs(j - playerX) <= 1)):
                if (explosions[i][j] != 0):
                    explosion_near_player = True

    return (explosion_near_player,)

def get_reachable_tiles(player_coords, field, bomb_power):
   """
   returns all tiles the player can reach in bomb_power
   """

   if bomb_power == 0:
       return []

   playerX, playerY = player_coords[0], player_coords[1]
   reachable_tiles = [(playerX, playerY)]
   
   # up
   for i in range(1, bomb_power+1):
       xCord = playerX
       yCord = playerY - i

		# check if tile is empty
       if(field[xCord][yCord] == 0):
           reachable_tiles.append((xCord, yCord))
           reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power-1)
       else:
           break
       
	# down
   for i in range(1, bomb_power+1):
       xCord = playerX
       yCord = playerY + i

		# check if tile is empty
       if(field[xCord][yCord] == 0):
           reachable_tiles.append((xCord, yCord))
           reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power-1)
       else:
           break
       
	# left
   for i in range(1, bomb_power+1):
       xCord = playerX - i
       yCord = playerY

		# check if tile is empty
       if(field[xCord][yCord] == 0):
           reachable_tiles.append((xCord, yCord))
           reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power-1)
       else:
           break
       
	# right
   for i in range(1, bomb_power+1):
       xCord = playerX + i
       yCord = playerY

		# check if tile is empty
       if(field[xCord][yCord] == 0):
           reachable_tiles.append((xCord, yCord))
           reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power-1)
       else:
           break
   
	# Do this for duplicate removal
   reachable_tiles = list( dict.fromkeys(reachable_tiles) )

   return reachable_tiles

def cord_is_valid(x,y):
    if x < 1 or x > ARENA_LENGTH - 2 or y < 1 or y > ARENA_WIDTH:
        return False
    else:
        return True
    
def check_for_neaby_explosion(player, explosions):
    playerX, playerY = get_player_coordinates(player)
    
    explosion_nearby = (0,)
    
    sizeX, sizeY = explosions.shape
    for i in range(sizeX):
        for j in range(sizeY):
            if(explosions[i][j] != 0):
                if(abs(playerX - i) + abs(playerY - j) <= 1):
                    explosion_nearby = (1,)
                
    return explosion_nearby

def get_safe_bomb_drop(player, field):
    # Player has no available bomb
    if not player[2]:
        return (False,)
    
    potential_bomb_coordinates = []
    potential_bomb_coordinates.append((get_player_coordinates(player),1))
    
    direction = get_direction_for_safe_tile(potential_bomb_coordinates, player, field)
    
	# If there is no way out
    if direction == list_of_steps.NODIR.value:
        return (False,)
    else:
        return (True,)