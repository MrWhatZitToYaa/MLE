from enum import Enum

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_OF_ACTIONS = len(ACTIONS)
PROBABILITIES_FOR_ACTIONS = [.2, .2, .2, .2, .2, .0]

# arena is 16 x 16 and each pixel can be occupied by 9 possible blocks
ARENA_LENGTH = 17
ARENA_WIDTH = 17
ARENA_SIZE = ARENA_LENGTH * ARENA_WIDTH
NUMBER_OF_BLOCKS = 11

NUM_OF_STATES = ARENA_SIZE * NUMBER_OF_BLOCKS

class list_of_blocks(Enum):
    EMPTY = 		0
    BRICK = 		1
    CRATE = 		2
    COIN =			3
    ENEMY =			4
    PLAYER = 		5
    EXPLOSION =		6
    SMOKE = 		7
    BOMB_TICK0 =	8
    BOMB_TICK1 =	9
    BOMB_TICK2 =	10

# General model parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.5

# Exploration probablility
EXPLORATION_DECAY_ACTIVE = True
EPSILON_DECAY = 0.99
DECAY_AFTER_ROUNDS = 10
STARTING_EXPLORATION_PROBABILITY = 1.0

# Custom Events
COIN_DIST_DECREASED = 'COIN_DIST_DECREASED'
