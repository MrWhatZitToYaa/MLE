import events as event
from typing import List
from collections import namedtuple, deque
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from .definitions import *
from .callbacks import state_to_features
from .state_to_feature_helpers import *
from .customEventAppender import appendCustomEvents

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=MAX_LEN_TRANSITIONS)
    self.model.train()
    """hyperparameters = [self.model.learning_rate,
                       self.model.discount_factor,
                       self.model.exploration_prob,
                       self.model.decay_active,
                       self.model.epsilon_decay,
                       self.model.epsilon_decay_after_rounds]

    # Store Hyperparameters
    with open("./monitor_training/hyperparameters.pkl", "wb") as file:
        pickle.dump(hyperparameters, file)"""
    self.total_reward = 0
    self.total_rewards = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    appendCustomEvents(self, events, new_game_state, old_game_state)

    #action = self.model.forward(state_to_features(new_game_state))
    train_step(self, old_game_state, self_action, new_game_state, reward_from_events(self, events))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # Store rewards
    # Add rewards of final steps, since there is no training

    self.total_reward += reward_from_events(self, events)
    self.total_rewards.append(self.total_reward)
    with open("./monitor_training/total_rewards.pkl", "wb") as file:
        pickle.dump(self.total_rewards, file)
    # Set total rewards to 0 for next round
    self.total_reward = 0
    '''
    # Store qTable size
    with open("./monitor_training/qTableSize.pkl", "wb") as file:
        pickle.dump(self.total_qTable_size, file)
    
    # Store exploration probability
    self.exploration_Probabilities.append(self.model.exploration_prob)
    with open("./monitor_training/exploration_probability.pkl", "wb") as file:
        pickle.dump(self.exploration_Probabilities, file)
    '''
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


def reward_from_events(self, event_sequence: List[str]) -> int:
    """
    Returns total reward for a sequence of events
    """
    rewards = {
        event.MOVED_LEFT: -5,
        event.MOVED_RIGHT: -5,
        event.MOVED_UP: -5,
        event.MOVED_DOWN: -5,
        event.WAITED: -5,
        event.INVALID_ACTION: -20,

        event.BOMB_DROPPED: -5,
        event.BOMB_EXPLODED: 0,

        event.CRATE_DESTROYED: 100,
        event.COIN_FOUND: 30,
        event.COIN_COLLECTED: 300,

        event.KILLED_OPPONENT: 0,
        event.KILLED_SELF: -600,

        event.GOT_KILLED: -300,
        event.OPPONENT_ELIMINATED: 200,
        event.SURVIVED_ROUND: 100,

        # Custom events

        # Collect coins
        # COIN_DIST_DECREASED: 5,

        # BOMB_DIST_INCREASED: 10,

        # Blow up Crates
        # STAYED_WITHIN_EXPLOSION_RADIUS: 0,
        MOVED_IN_SAFE_DIRECTION: 20,
        # GOT_OUT_OF_EXPLOSION_RADIUS: 20,
        # DROPPED_BOMB_WITH_NO_WAY_OUT: -100,
        SURVIVED_EXPLOSION: 10,
        # WALKED_INTO_EXPLOSION: -50,

        # General Movement
        # VISITED_SAME_PLACE: -20
    }

    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]

    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward


def train_step(self, old_state, action, new_state, reward):
    if action is not None:
        action_select = torch.zeros(len(ACTIONS), dtype=torch.int64)
        action_select[ACTIONS.index(action)] = 1

        state_action_value = torch.masked_select(self.model.forward(old_state), action_select.bool())
        next_state_action_value = self.model.forward(new_state).max().unsqueeze(0)
        expected_state_action_value = (next_state_action_value * LEARNING_RATE) + reward

        loss = self.model.criterion(state_action_value, expected_state_action_value)

        with open("loss_log.txt", "a") as loss_log:
            loss_log.write(str(loss.item()) + "\t")
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

