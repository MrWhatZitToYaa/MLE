import numpy as np
import settings as s
import os
import random
import pickle
import torch.nn.functional as torch_functions

import definitions as d
import model as m

def setup(self):
    """
    Copy pasted setup method
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(d.ACTIONS))
        # TODO: diese Funktion implementieren
        self.model = m.QLearning(d.NUM_OF_STATES, d.NUM_OF_ACTIONS, d.learning_rate, d.discount_factor)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    if self.train and random.random() < d.exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(d.ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    state = state_to_features(game_state)
    next_action = self.model.get_action_for_state(state)
    return next_action

def state_to_features(game_state: dict) -> int:
    """
    Converts the game state into a number
    :param game_state:  A dictionary describing the current game board.
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
    
    round = [round]
    step = [step]
    field = field.flatten().tolist()
    bombs = np.array([np.array([i[0][0], i[0][1], i[1]]) for i in bombs]).flatten().tolist()
    explosions = explosions.flatten().tolist()
    coins = np.array([np.array([i[0], i[1]]) for i in coins]).flatten().tolist()
    player = [player[0], player[1], player[2], player[3][0],player[3][1]]
    enemies = [np.array([i[0], i[1], i[2], i[3][0],i[3][1]]) for i in enemies]
    enemies_flat = []
    for sublist in enemies:
         for item in sublist:
              enemies_flat.append(item)
              
    total_list = round + step + field + bombs + explosions + coins + player + enemies_flat 
    
	#to difficult for now maybe even unessecary, keep for later
    """
    arena = np.zeros((d.ARENA_LENGTH, d.ARENA_WIDTH))

    round = game_state["round"]
    step = game_state["step"]
    
    field = [d.list_of_blocks.EMPTY.value if x == 0 else 
             d.list_of_blocks.BRICK.value if x == -1 else
             d.list_of_blocks.CRATE.value
             for x in field]
    bombs = game_state["bombs"]
    explosions = game_state["explosion_map"]
    coins = game_state["coins"]
    player = game_state["self"]
    enemies = game_state["others"]
    
    for i in range(0,d.ARENA_LENGTH):
         for j in range(0,d.ARENA_WIDTH):
              arena[i][j] = field[i][j]
    """
    
    return total_list
