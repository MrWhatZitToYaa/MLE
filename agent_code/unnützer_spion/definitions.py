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
NUM_FEATURES = 5

class list_of_steps(Enum):
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
    NUMBER_OF_BLOCKS = 14

# General model parameters
# best settings so far: LEARNING_RATE = 0.00005, WEIGHT_DECAY = 2e-5, StFV14

INPUT_CHANNELS = ARENA_SIZE*NUM_FEATURES
FC1 = 256
FC2 = 256
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 2e-5

#Agent Hyperparameters
GAMMA = 0.99
MAX_MEM_SIZE = 50000
BATCH_SIZE = 32
MAX_LEN_TRANSITIONS = MAX_MEM_SIZE
TARGET_UPDATE_FREQ = 100


# Exploration probablility
EXPLORATION_DECAY_ACTIVE = True
EPSILON_DECAY = 0.95
DECAY_AFTER_ROUNDS = 10
STARTING_EXPLORATION_PROBABILITY = 1#0.035


# Custom Events
COIN_DIST_DECREASED = 'COIN_DIST_DECREASED'
BOMB_DIST_INCREASED = 'BOMB_DIST_INCREASED'
STAYED_WITHIN_EXPLOSION_RADIUS = 'STAYED_WITHIN_EXPLOSION_RADIUS'
MOVED_IN_SAFE_DIRECTION = 'MOVED_IN_SAFE_DIRECTION'
GOT_OUT_OF_EXPLOSION_RADIUS = 'GOT_OUT_OF_EXPLOSION_RADIUS'
DROPPED_BOMB_WITH_NO_WAY_OUT = 'DROPPED_BOMB_WITH_NO_WAY_OUT'
VISITED_SAME_PLACE = 'VISITED_SAME_PLACE'
SURVIVED_EXPLOSION = 'SURVIVED_EXPLOSION'
WALKED_INTO_EXPLOSION = 'WALKED_INTO_EXPLOSION'

