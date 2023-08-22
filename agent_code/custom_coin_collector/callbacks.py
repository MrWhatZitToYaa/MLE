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
    # probabilistic choice of action
	# TODO: code line aus anderem Github repo kopiert, funktion muss noch verstanden werden und bissel abändern oder löschen falls nicht verwedent!
    #next_action = np.random.choice(ACTIONS, p=torch_functions.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    
	# choose action with maximum reward
    next_action = d.ACTIONS[np.argmax(self.model.get_values_for_state(game_state))]
    
    return next_action

def state_to_features(game_state: dict) -> int:
    """
    Converts the game state into a number
    TODO: Implement this using dictionary or something

    :param game_state:  A dictionary describing the current game board.
    :return: int
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    return 1
