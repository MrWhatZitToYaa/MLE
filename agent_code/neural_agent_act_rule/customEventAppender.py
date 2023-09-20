import events as event
from .definitions import *
from .state_to_feature_helpers import *


def appendCustomEvents(self, events, new_game_state, old_game_state):
    # Multiplayer approved
    if is_coin_dist_decreased(old_game_state, new_game_state):
        events.append(COIN_DIST_DECREASED)
        self.logger.debug(f'Custom event occurred: {COIN_DIST_DECREASED}')

    # Multiplayer semi-approved, may need modification if performance is bad
    if took_step_safe_direction(old_game_state, new_game_state):
        events.append(MOVED_IN_SAFE_DIRECTION)
        self.logger.debug(f'Custom event occurred: {MOVED_IN_SAFE_DIRECTION}')

    # Multiplayer approved
    if took_step_to_safe_tile(old_game_state, new_game_state):
        events.append(MOVED_CLOSER_TO_SAVE_TILE)
        self.logger.debug(f'Custom event occurred: {MOVED_CLOSER_TO_SAVE_TILE}')

    # Multiplayer approved
    if took_step_away_from_safe_tile(old_game_state, new_game_state):
        events.append(MOVED_AWAY_FROM_SAVE_TILE)
        self.logger.debug(f'Custom event occurred: {MOVED_AWAY_FROM_SAVE_TILE}')

    # Multiplayer approved
    if got_out_of_explosion_radius(old_game_state, new_game_state):
        events.append(GOT_OUT_OF_EXPLOSION_RADIUS)
        self.logger.debug(f'Custom event occurred: {GOT_OUT_OF_EXPLOSION_RADIUS}')

    # Multiplayer semi-approved, if no problems then fine, but not 100% sure if killed-self is agent
    # specific event or not. If yes (what I hope, then fine)
    if did_not_walk_into_explosion(old_game_state, events):
        events.append(SURVIVED_EXPLOSION)
        self.logger.debug(f'Custom event occurred: {SURVIVED_EXPLOSION}')

    # Multiplayer semi-approved, if no problems then fine, but not 100% sure if killed-self is agent
    # specific event or not. If yes (what I hope, then fine)
    if stayed_in_explosion_radius(old_game_state, new_game_state):
        events.append(STAYED_IN_EXPLOSION_RADIUS)
        self.logger.debug(f'Custom event occurred: {STAYED_IN_EXPLOSION_RADIUS}')

    # Multiplayer approved
    if dropped_bomb_near_crate(old_game_state, new_game_state):
        events.append(DROPPED_BOMB_NEAR_CRATE)
        self.logger.debug(f'Custom event occurred: {DROPPED_BOMB_NEAR_CRATE}')
    """
    if run_away_to_saftey_if_on_top_of_bomb(old_game_state, new_game_state):
         events.append(RUN_AWAY_FROM_BOMB_IF_ON_TOP)
         self.logger.debug(f'Custom event occurred: {RUN_AWAY_FROM_BOMB_IF_ON_TOP}')
    """

    return events


def is_coin_dist_decreased(old_state, new_state):
    """
    Checks whether the agent moved towards a coin.
    """
    old_min_d = find_min_coin_distance(old_state["coins"], *old_state["self"][3])
    new_min_d = find_min_coin_distance(new_state["coins"], *new_state["self"][3])

    return new_min_d < old_min_d


def took_step_safe_direction(old_state, new_state):
    """
    Checks whether the agent moved away from a dangerous bomb.
    """
    bomb = find_closest_dangerous_bomb(old_state["bombs"], old_state["field"], old_state["self"][3])
    if bomb == None:
        return False
    old_dist = np.linalg.norm(
        np.array((old_state["self"][3][0], old_state["self"][3][1])) - np.array((bomb[0][0], bomb[0][1])))
    new_dist = np.linalg.norm(
        np.array((new_state["self"][3][0], new_state["self"][3][1])) - np.array((bomb[0][0], bomb[0][1])))

    return old_dist < new_dist


def stayed_within_explosion_radius(old_state, new_state):
    """
    Checks whether the agent continues to be in the radius of an explosion.
    """
    if not within_explosion_radius(old_state["self"][0][0], old_state["self"][0][1], old_state["field"],
                                   old_state["bombs"]):
        return False
    else:
        if within_explosion_radius(new_state["self"][0][0], new_state["self"][0][1], new_state["field"],
                                   new_state["bombs"]):
            return True
        else:
            return False


def got_out_of_explosion_radius(old_state, new_state):
    if not within_explosion_radius(old_state["self"], old_state["field"], old_state["bombs"]):
        return False
    else:
        if within_explosion_radius(new_state["self"], new_state["field"], new_state["bombs"]):
            return False
        else:
            return True


def reachable_safe_tile_exists(player_coords, field, bombs):
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, player_coords)
    if dangerous_bomb == None:
        return None
    radius = get_blast_coords(dangerous_bomb[0], field)
    reachable_tiles = get_reachable_tiles(player_coords, field)

    for tile in reachable_tiles:
        if tile not in radius:
            return True
    else:
        return False


def check_if_survived_explosion(old_state, new_state):
    """
    Checks if an explosion was survuved. MAY NOT WORK RELIABLY WITH MULTIPLE AGENTS!
    """
    if np.sum(old_state["explosion_map"]) != 0 and np.sum(new_state["explosion_map"]) == 0:
        return True
    else:
        return False


def did_not_walk_into_explosion(old_state, events):
    area = get_area_around_player(old_state["field"],
                                  old_state["explosion_map"],
                                  old_state["self"],
                                  old_state["bombs"],
                                  old_state["others"])
    explosion_was_nearby = False

    for i in area:
        if (i == list_of_blocks.EXPLOSION0.value):
            explosion_was_nearby = True

    for i in events:
        if (i == 'KILLED_SELF' and explosion_was_nearby):
            return False

    return True


def stayed_in_explosion_radius(old_state, events):
    area = get_area_around_player(old_state["field"],
                                  old_state["explosion_map"],
                                  old_state["self"],
                                  old_state["bombs"],
                                  old_state["others"])
    explosion_was_nearby = False

    for i in area:
        if (i == list_of_blocks.EXPLOSION0.value):
            explosion_was_nearby = True

    for i in events:
        if (i == 'KILLED_SELF' and explosion_was_nearby):
            return True

    return False


def walked_into_explosion(new_state):
    new_coords = get_agent_coordinates(new_state["self"])

    if new_state["explosion_map"][new_coords[0]][new_coords[1]] != 0:
        return True
    else:
        return False


def dropped_bomb_near_crate(old_game_state, new_game_state):
    area = get_area_around_player(old_game_state["field"],
                                  old_game_state["explosion_map"],
                                  old_game_state["self"],
                                  old_game_state["bombs"],
                                  old_game_state["others"])
    nearCrate = False
    # Bomb immedeately imminent to agent
    for i in area:
        if (i == list_of_blocks.CRATE.value):
            nearCrate = True

    # Bomb was already dropped
    playerX, playerY = get_agent_coordinates(new_game_state["self"])
    for i in old_game_state["bombs"]:
        bombX = i[0][0]
        bombY = i[0][1]

        if bombX == playerX and bombY == playerY:
            return False

    if (nearCrate):
        for i in new_game_state["bombs"]:
            bombX = i[0][0]
            bombY = i[0][1]

            if bombX == playerX and bombY == playerY:
                return True

    return False


def took_step_to_safe_tile(old_game_state, new_game_state):
    direction_safty = get_direction_for_safe_tile(old_game_state["bombs"], old_game_state["self"],
                                                  old_game_state["field"])

    player_X_new, player_Y_new = get_agent_coordinates(new_game_state["self"])
    direction_taken = get_direction_for_object(player_X_new, player_Y_new, old_game_state["self"],
                                               old_game_state["field"])

    if (direction_safty == direction_taken):
        return True
    else:
        return False


def took_step_away_from_safe_tile(old_game_state, new_game_state):
    direction_safty = get_direction_for_safe_tile(old_game_state["bombs"], old_game_state["self"],
                                                  old_game_state["field"])

    player_X_new, player_Y_new = get_agent_coordinates(new_game_state["self"])
    direction_taken = get_direction_for_object(player_X_new, player_Y_new, old_game_state["self"],
                                               old_game_state["field"])

    if (direction_safty != direction_taken and direction_safty != list_of_steps.NO_TARGET.value):
        return True
    else:
        return False