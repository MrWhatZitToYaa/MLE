import events as event
from typing import List
from collections import namedtuple, deque
import pickle
import torch
import os
import torch.nn.functional as F
import numpy as np
from .definitions import *
from .callbacks import state_to_features, act_rule
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
    self.total_reward = 0
    self.total_rewards = []
    self.transitions = deque(maxlen=MAX_LEN_TRANSITIONS)
    self.model.train()


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

    train_step(self, old_game_state, self_action)


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
        event.INVALID_ACTION: -40,

        event.BOMB_DROPPED: -20,
        event.BOMB_EXPLODED: 0,

        event.CRATE_DESTROYED: 125,
        event.COIN_FOUND: 30,
        event.COIN_COLLECTED: 150,

        event.KILLED_OPPONENT: 200,
        event.KILLED_SELF: -600,

        event.GOT_KILLED: -300,
        event.OPPONENT_ELIMINATED: 200,
        event.SURVIVED_ROUND: 100,

        # Blow up Crates
        DROPPED_BOMB_NEAR_CRATE: 30,
        SURVIVED_EXPLOSION: 5,
        STAYED_IN_EXPLOSION_RADIUS: -5,

        # Safety
        MOVED_CLOSER_TO_SAVE_TILE: 10,
        MOVED_AWAY_FROM_SAVE_TILE: -10
    }

    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]

    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward


def train_step(self, old_state, action):
    if action is not None and act_rule(self, old_state) is not None:
        old_state_action_value = self.model.forward(old_state).unsqueeze(0)
        # expected action is what a rule based agent would have done
        rule_based_action = act_rule(self, old_state)
        expected_state_action_value = torch.tensor(ACTIONS.index(rule_based_action), dtype=torch.long).unsqueeze(0)

        # log loss for plotting later
        loss = self.model.criterion(old_state_action_value, expected_state_action_value)
        with open("loss_log.txt", "a") as loss_log:
            loss_log.write(str(loss.item()) + "\t")

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
