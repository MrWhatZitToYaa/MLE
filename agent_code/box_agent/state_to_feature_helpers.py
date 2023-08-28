import numpy as np

from .definitions import *

def get_player_coordinates(player):
    playerX = player[3][0]
    playerY = player[3][1]
    
    return (playerX, playerY)

def get_safe_tiles(reachable_tiles, blast_coords):
    safe_tiles = []
    for tile in reachable_tiles:
            if tile not in blast_coords:
                safe_tiles.append(tile)

    return safe_tiles

def get_area_around_player(field, player):
    field = field.flatten()
    # Transform arena
    transformed_field = [list_of_blocks.EMPTY.value if x == 0 else 
                        list_of_blocks.BRICK.value if x == -1 else
                        list_of_blocks.CRATE.value
                        for x in field]
    
    playerX, playerY = get_player_coordinates(player)

	# Init area, only use odd sizes
    area_around_player_size = 3
    area_around_player = np.zeros((area_around_player_size, area_around_player_size))
    
    #Copy area around player
    for i in range(0,area_around_player_size):
         for j in range(0,area_around_player_size):
              area_around_player[i][j] = transformed_field[(i-1+playerY)*ARENA_LENGTH + j-1 + playerX]
         
	# Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value
    
	# Add area_around_player to feature_vector
    return tuple(area_around_player.flatten())

def find_min_coin_coordinate(coins: list, player):
    playerX, playerY = get_player_coordinates(player)

    min_x = ARENA_WIDTH
    min_y = ARENA_LENGTH
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (coinX, coinY) in coins:
        d = np.linalg.norm(np.array((playerX, playerY)) - np.array((coinX, coinY)))
        if d < min_d:
            min_d = d
            min_x = coinX - playerX
            min_y = coinY - playerY
    return (min_x, min_y)

def find_min_coin_distance(coins: list, playerX, playerY):
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (coinX, coinY) in coins:
        d = np.linalg.norm(np.array((playerX, playerY)) - np.array((coinX, coinY)))
        if d < min_d:
            min_d = d
    return min_d

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

def within_explosion_radius(playerX, playerY, field, bombs):
    """
    Checks whether the agent is within the radius of a bomb. It doesn't take into account when the bomb is set to go off.
    """
    player_coords = (playerX, playerY)
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
        d = np.linalg.norm(np.array((player_coords[0], player_coords[1])) - np.array((bomb[0][0], bomb[0][1])))
        if d < min_d:
            min_d = d
            closest_dangerous_bomb = bomb
    return closest_dangerous_bomb


def get_reachable_tiles(player_coords, field):
    x, y = player_coords[0], player_coords[1]
    reachable_tiles = [(x, y)]
    bomb_power = 3

    # y
    for i in range(1, bomb_power):
        # both pos and neg y
        for m in range(2):
            if m == 0: current_y = y+i
            else: current_y = y-1

            if current_y >= ARENA_WIDTH: 
                break
            elif field[x][current_y] != 0:
                break        
            reachable_tiles.append((x, current_y))

            for j in range(1, bomb_power - 1):
                if x+j >= ARENA_LENGTH: break
                elif field[x+j][current_y] != 0:
                    break
                reachable_tiles.append((x+j, current_y))

            for k in range(1, bomb_power -1):
                if x-k >= ARENA_LENGTH: break
                elif field[x-k][current_y] != 0:
                    break
                reachable_tiles.append((x-k, current_y))        

    # x
    for i in range(1, bomb_power):
        # both pos and neg x
        for m in range(2):
            if m == 0: current_x = x+i
            else: current_x = x-1

        if current_x >= ARENA_LENGTH: 
            break
        elif field[current_x][y] != 0:
            break        
        reachable_tiles.append((current_x, y))

        for j in range(1, bomb_power-1):
            if y+j >= ARENA_WIDTH: break
            elif field[current_x][y+j] != 0:
                break
            reachable_tiles.append((current_x, y+j))

        for k in range(1, bomb_power-1):
            if y-k >= ARENA_WIDTH: break
            elif field[current_x][y-k] != 0:
                break
            reachable_tiles.append((current_x, y-k)) 

    return reachable_tiles