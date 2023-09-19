from enum import Enum

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_OF_ACTIONS = len(ACTIONS)
#PROBABILITIES_FOR_ACTIONS = [0.2, 0.2, 0.2, 0.2, 0.2, 0]
PROBABILITIES_FOR_ACTIONS = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
#PROBABILITIES_FOR_ACTIONS = [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]

# arena is 15 x 15 and each pixel can be occupied by 14 possible blocks
ARENA_LENGTH = 17
ARENA_WIDTH = 17
ARENA_SIZE = ARENA_LENGTH * ARENA_WIDTH

class list_of_steps(Enum):
    NO_TARGET = -1
    UP =         0
    DOWN =       1
    LEFT =       2
    RIGHT =      3
    NODIR = 	 4

class list_of_blocks(Enum):
    EMPTY =         0
    BRICK =         1
    CRATE =         2
    COIN =          3
    ENEMY =         4
    PLAYER =        5
    EXPLOSION0 =    6
    EXPLOSION1 =    7
    EXPLOSION2 =    8
    SMOKE =         9
    BOMB_TICK0 =    10
    BOMB_TICK1 =    11
    BOMB_TICK2 =    12
    BOMB_TICK3 =    13
    DANGER	   =	14
    NUMBER_OF_BLOCKS = 15

# General model parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.7
# Has to be at least one
NUMBER_OF_RELEVANT_STATES = 6

# Exploration probablility
EXPLORATION_DECAY_ACTIVE = True
EPSILON_DECAY = 0.997
DECAY_AFTER_ROUNDS = 10
STARTING_EXPLORATION_PROBABILITY = 1.0

# Custom Events
COIN_DIST_DECREASED = 'COIN_DIST_DECREASED'
STAYED_WITHIN_EXPLOSION_RADIUS = 'STAYED_WITHIN_EXPLOSION_RADIUS'
MOVED_IN_SAFE_DIRECTION = 'MOVED_IN_SAFE_DIRECTION'
GOT_OUT_OF_EXPLOSION_RADIUS = 'GOT_OUT_OF_EXPLOSION_RADIUS'
SURVIVED_EXPLOSION = 'SURVIVED_EXPLOSION'
STAYED_IN_EXPLOSION_RADIUS = 'STAYED_IN_EXPLOSION_RADIUS'
DROPPED_BOMB_NEAR_CRATE = 'DROPPED_BOMB_NEAR_CRATE'
RUN_AWAY_FROM_BOMB_IF_ON_TOP = 'RUN_AWAY_FROM_BOMB_IF_ON_TOP'
MOVED_CLOSER_TO_SAVE_TILE = 'MOVED_CLOSER_TO_SAVE_TILE'
MOVED_AWAY_FROM_SAVE_TILE = 'MOVED_AWAY_FROM_SAVE_TILE'
