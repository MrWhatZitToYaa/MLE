import numpy as np
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
    self.count = 0
    if self.train or not os.path.isfile("my-saved-model.ptt"):
        self.logger.info("Setting up model from scratch.")
        self.Q_eval = DQN(INPUT_CHANNELS, FC1, FC2, NUM_OF_ACTIONS)
        self.Q_target = DQN(INPUT_CHANNELS, FC1, FC2, NUM_OF_ACTIONS)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q_eval = pickle.load(file)
            
	# Keeps track of the scores if evaluation mode is active
    self.scores = []


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    self.count += 1
    exploration_prob = STARTING_EXPLORATION_PROBABILITY
    if self.count % DECAY_AFTER_ROUNDS == 0:
        exploration_prob *= EPSILON_DECAY

    if self.train and random.random() < exploration_prob:
        return np.random.choice(ACTIONS, p=PROBABILITIES_FOR_ACTIONS)
    self.logger.debug("Querying model for action.")
    res = np.random.choice(ACTIONS, p=F.softmax(self.Q_eval.forward(game_state), dim=0).detach().numpy())
    #print('RESULT:', res)
    return res
