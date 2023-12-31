import numpy as np
from scipy.spatial.distance import cdist

from .definitions import *
def get_player_coordinates(player):
    playerX = player[3][0]
    playerY = player[3][1]

    return (playerX, playerY)


def get_safe_tiles_from_specific_bomb(reachable_tiles, blast_coords):
    safe_tiles = []
    for tile in reachable_tiles:
        if tile not in blast_coords:
            safe_tiles.append(tile)

    return safe_tiles


def get_safe_tiles(reachable_tiles, bombs, field):
    safe_tiles = []
    if bombs == []:
        return reachable_tiles
    else:
        for bomb in bombs:
            for tile in reachable_tiles:
                if tile not in get_blast_coords(bomb[0], field):
                    safe_tiles.append(tile)
    return safe_tiles


def get_direction_to_closetes_safe_tile(player, safe_tiles):
    closest_safe_tileX, closest_safe_tileY = find_min_safe_tile_relative_coordinate(safe_tiles, player)
    return get_direction_for_object(closest_safe_tileX, closest_safe_tileY)


def get_area_around_player(field, explosions, player):
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

    # Copy area around player
    for i in range(0, area_around_player_size):
        for j in range(0, area_around_player_size):
            area_around_player[i][j] = transformed_field[i - 1 + playerX][j - 1 + playerY]

    # Add player to area
    area_around_player[1][1] = list_of_blocks.PLAYER.value

    # Add explosions to area
    sizeX, sizeY = explosions.shape
    for x in range(sizeX):
        for y in range(sizeY):
            if (explosions[x][y] != 0):
                if (abs(playerX - x) + abs(playerY - y) <= 1):
                    explosion_rel_X = x - playerX
                    explosion_rel_Y = y - playerY
                    area_around_player[explosion_rel_X + 1][explosion_rel_Y + 1] = list_of_blocks.EXPLOSION0.value

    # Add area_around_player to feature_vector
    return tuple(area_around_player.flatten())


def find_min_coin_relative_coordinate(coins: list, player):
    playerX, playerY = get_player_coordinates(player)

    min_x = ARENA_WIDTH
    min_y = ARENA_LENGTH
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (coinX, coinY) in coins:
        d = abs(coinX - playerX) + abs(coinY - playerY)
        if d < min_d:
            min_d = d
            min_x = coinX - playerX
            min_y = coinY - playerY
    return (min_x, min_y)


def find_min_safe_tile_relative_coordinate(safe_tiles, player):
    playerX, playerY = get_player_coordinates(player)

    min_x = ARENA_WIDTH
    min_y = ARENA_LENGTH
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (tileX, tileY) in safe_tiles:
        d = abs(tileX - playerX) + abs(tileY - playerY)
        if d < min_d:
            min_d = d
            min_x = tileX - playerX
            min_y = tileY - playerY
    return (min_x, min_y)


def find_min_coin_distance(coins: list, playerX, playerY):
    min_d = ARENA_WIDTH + ARENA_LENGTH
    for (coinX, coinY) in coins:
        # d = np.linalg.norm(np.array((playerX, playerY)) - np.array((coinX, coinY)))
        # d = cdist(np.array(playerX, playerY), np.array(coinX, coinY), 'cityblock')
        # Manhattan dist
        player_arr = np.array((playerX, playerY))
        coin_arr = np.array((coinX, coinY))
        d = sum(abs(player_arr - coin_arr) for player_arr, coin_arr in zip(player_arr, coin_arr))
        if d < min_d:
            min_d = d
    return min_d


def get_direction_for_object(objectX, objectY):
    directionX = -2
    directionY = -2

    if (objectX < 0):
        directionX = 1
    if (objectX == 0):
        directionX = 0
    if (objectX > 0):
        directionX = -1

    if (objectY < 0):
        directionY = 1
    if (objectY == 0):
        directionY = 0
    if (objectY > 0):
        directionY = -1

    return (directionX, directionY)


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
        # d = np.linalg.norm(np.array((player_coords[0], player_coords[1])) - np.array((bomb[0][0], bomb[0][1])))
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
            if ((abs(i - playerY) <= 1) and (abs(j - playerX) <= 1)):
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
    for i in range(1, bomb_power + 1):
        xCord = playerX
        yCord = playerY - i

        # check if tile is empty
        if (field[xCord][yCord] == 0):
            reachable_tiles.append((xCord, yCord))
            reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power - 1)
        else:
            break

    # down
    for i in range(1, bomb_power + 1):
        xCord = playerX
        yCord = playerY + i

        # check if tile is empty
        if (field[xCord][yCord] == 0):
            reachable_tiles.append((xCord, yCord))
            reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power - 1)
        else:
            break

    # left
    for i in range(1, bomb_power + 1):
        xCord = playerX - i
        yCord = playerY

        # check if tile is empty
        if (field[xCord][yCord] == 0):
            reachable_tiles.append((xCord, yCord))
            reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power - 1)
        else:
            break

    # right
    for i in range(1, bomb_power + 1):
        xCord = playerX + i
        yCord = playerY

        # check if tile is empty
        if (field[xCord][yCord] == 0):
            reachable_tiles.append((xCord, yCord))
            reachable_tiles += get_reachable_tiles((xCord, yCord), field, bomb_power - 1)
        else:
            break

    # Do this for duplicate removal
    reachable_tiles = list(dict.fromkeys(reachable_tiles))

    return reachable_tiles


def cord_is_valid(x, y):
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
            if (explosions[i][j] != 0):
                if (abs(playerX - i) + abs(playerY - j) <= 1):
                    explosion_nearby = (1,)

    return explosion_nearby