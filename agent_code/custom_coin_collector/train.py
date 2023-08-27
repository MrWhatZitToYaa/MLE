import events as event
from typing import List
from collections import namedtuple, deque
import pickle
from .definitions import *
from .callbacks import state_to_features
from .state_to_feature import find_min_distance

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
    """
    Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)
    """
    if is_coin_dist_decreased(old_game_state, new_game_state):
        events.append(COIN_DIST_DECREASED)
        self.logger.debug(f'Custom event occurred: {COIN_DIST_DECREASED}')


    # state_to_features is defined in callbacks.py
    self.model.train(state_to_features(old_game_state),
                     self_action,
                     reward_from_events(self, events),
                     state_to_features(new_game_state),
                     old_game_state["round"])
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

	# Add total rewards for this round to a file

	# Add rewards of final steps, since there is no training
    self.model.total_reward += reward_from_events(self, events)
    self.total_rewards.append(self.model.total_reward)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    with open("./rewards/total_rewards.pkl", "wb") as file:
        pickle.dump(self.total_rewards, file)
        
	# set total rewards to 0 for next round
    self.total_reward = 0

def reward_from_events(self, event_sequence: List[str]) -> int:
    """
    Returns total reward for a sequence of events
    """
    rewards = {
        event.MOVED_LEFT: -5,
        event.MOVED_RIGHT: -5,
        event.MOVED_UP: -5,
        event.MOVED_DOWN: -5,
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

        COIN_DIST_DECREASED: 5
    }
    
    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]
            
    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward

def is_coin_dist_decreased(old_state, new_state):
    """
    Checks whether the agent moved towards a coin.
    """
    old_min_d = find_min_distance(old_state["coins"], *old_state["self"][3])
    new_min_d = find_min_distance(new_state["coins"], *new_state["self"][3])

    return new_min_d < old_min_d