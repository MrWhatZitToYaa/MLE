from enum import Enum

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_OF_ACTIONS = len(ACTIONS)
PROBABILITIES_FOR_ACTIONS = [.2, .2, .2, .2, .1, .1]

# arena is 16 x 16 and each pixel can be occupied by 9 possible blocks
ARENA_LENGTH = 16
ARENA_WIDTH = 16
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

learning_rate = 0.1
discount_factor = 0.99
EXPLORATION_PROBABIOLITY = 0.2