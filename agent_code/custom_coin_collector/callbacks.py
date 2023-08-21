import numpy as np
import settings as s
import os
import random
import pickle
import torch.nn.functional as torch_functions

import definitions as d

def setup(self):
    """
    Copy pasted setup method
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(d.ACTIONS))
        # TODO: diese Funktion implementieren
        self.model = simple_QLearn()
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
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(d.ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # probabilistic choice of action
	# TODO: code line aus anderem Github repo kopiert, funktion muss noch verstanden werden und bissel abändern oder löschen falls nicht verwedent!
    #next_action = np.random.choice(ACTIONS, p=torch_functions.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    
	# choose action with maximum reward
    next_action = d.ACTIONS[np.argmax(self.model.forward(game_state))]
