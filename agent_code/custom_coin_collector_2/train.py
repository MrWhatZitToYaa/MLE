import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as event
from .model import CustomModel

from .state_to_feature_helper import get_player_pos


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.epsilon = 0.02
    self.model = CustomModel()

def do_training_step(self, game_state: dict):
    if (random.uniform(0, 1) < self.epsilon):
        action = np.random.choice(self.model.actions, p=self.model.action_probabilities)
        # action = np.random.choice(self.model.actions) # do a random action
    else:
        actionIndex = self.model.predict_action(game_state)
        action = self.model.actions[actionIndex]

    #self.logger.debug(f'Epsilon:', self.epsilon)

    self.logger.debug(f'Action:', action)

    #reduce_epsilon(self)
    #self.logger.debug(f'Epsilon reduced to:', self.epsilon)

    return action


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    :param self: standard object that is passed to all methods
    :param old_game_state: The state that was passed to the last call of `act`
    :param self_action: The action taken by the agent
    :param new_game_state: The state the agent is in now
    :param events: Diff between old and new game_state
    """
    total_rewards = reward_from_events(self, events)
    self.model.update_qtable(old_game_state, new_game_state, self_action, total_rewards, False)

    # update position
    new_pos = get_player_pos(new_game_state)
    self.model.update_last_positions(new_pos)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards
    """
    rewards = reward_from_events(self, events)
    self.model.update_qtable(last_game_state, None, last_action, rewards, True)
    score = last_game_state['self'][1]

    print('Final score:', score)


def reduce_epsilon(self):
    self.epsilon *= 0.95

def reward_from_events(self, event_sequence: List[str]) -> int:
    """
    Returns total reward for a sequence of events
    """
    rewards = {
        event.MOVED_LEFT: -1,
        event.MOVED_RIGHT: -1,
        event.MOVED_UP: -1,
        event.MOVED_DOWN: -1,
        event.WAITED: -10,
        event.INVALID_ACTION: -20,
        
        event.BOMB_DROPPED: -100,
        event.BOMB_EXPLODED: 0,
        
        event.CRATE_DESTROYED: 5,
        event.COIN_FOUND: 10,
		event.COIN_COLLECTED: 100,
        
        event.KILLED_OPPONENT: 0,
        event.KILLED_SELF: -100,
        
		event.GOT_KILLED: -500,
        event.OPPONENT_ELIMINATED: 200,
		event.SURVIVED_ROUND: 50,
    }
    
    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]
            
    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward