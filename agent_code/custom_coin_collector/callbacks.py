import numpy as np
import settings as s
import os
import random
import pickle
import torch.nn.functional as torch_functions

from .definitions import *
from .model import *
from .state_to_feature import state_to_features

def setup(self):
    """
    Copy pasted setup method
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        # TODO: diese Funktion implementieren
        self.model = QLearning(NUM_OF_STATES, NUM_OF_ACTIONS, learning_rate, discount_factor, exploration_prob)
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
    if self.train and random.random() < exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=PROBABILITIES_FOR_ACTIONS)

    self.logger.debug("Querying model for action.")
    state = state_to_features(game_state)
    next_action = self.model.get_action_for_state(state)
    return next_action


