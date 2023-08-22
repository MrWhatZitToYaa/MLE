ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# arena is 15 x 15 and each pixel can be occupied by 9 possible blocks
NUM_OF_STATES = 15 * 15 * 9
NUM_OF_ACTIONS = len(ACTIONS)

learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0