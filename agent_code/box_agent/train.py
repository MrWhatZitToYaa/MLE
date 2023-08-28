import events as event
from typing import List
from collections import namedtuple, deque
import pickle
import numpy as np
from .definitions import *
from .callbacks import state_to_features
from .state_to_feature import find_min_coin_distance, within_explosion_radius, find_closest_dangerous_bomb, get_blast_coords, get_reachable_tiles

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
    self.total_qTable_size = []
    self.exploration_Probabilities = []
    
    hyperparameters = [self.model.learning_rate,
                       self.model.discount_factor,
                       self.model.exploration_prob,
                       self.model.decay_active,
                       self.model.epsilon_decay,
                       self.model.epsilon_decay_after_rounds]

    # Store Hyperparameters
    with open("./monitor_training/hyperparameters.pkl", "wb") as file:
        pickle.dump(hyperparameters, file)

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

    if stayed_within_explosion_radius(old_game_state, new_game_state):
        events.append(STAYED_WITHIN_EXPLOSION_RADIUS)
        self.logger.debug(f'Custom event occurred: {STAYED_WITHIN_EXPLOSION_RADIUS}')

    if took_step_safe_direction(old_game_state, new_game_state):
        events.append(MOVED_IN_SAFE_DIRECTION)
        self.logger.debug(f'Custom event occurred: {MOVED_IN_SAFE_DIRECTION}')
    
    if got_out_of_explosion_radius(old_game_state, new_game_state):
        events.append(GOT_OUT_OF_EXPLOSION_RADIUS)
        self.logger.debug(f'Custom event occurred: {GOT_OUT_OF_EXPLOSION_RADIUS}')

    if event.BOMB_DROPPED in events and not reachable_safe_tile_exists(new_game_state["self"][3], new_game_state["field"], new_game_state["bombs"]):        
        events.append(DROPPED_BOMB_WITH_NO_WAY_OUT)
        self.logger.debug(f'Custom event occurred: {DROPPED_BOMB_WITH_NO_WAY_OUT}')


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
    
	# Add qTable size
    self.total_qTable_size.append(len(self.model.q_table))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

	# Store rewards
	# Add rewards of final steps, since there is no training
    self.model.total_reward += reward_from_events(self, events)
    self.total_rewards.append(self.model.total_reward)
    with open("./monitor_training/total_rewards.pkl", "wb") as file:
        pickle.dump(self.total_rewards, file)
    # Set total rewards to 0 for next round
    self.model.total_reward = 0
        
	# Store qTable size
    with open("./monitor_training/qTableSize.pkl", "wb") as file:
        pickle.dump(self.total_qTable_size, file)
        
	# Store exploration probability
    self.exploration_Probabilities.append(self.model.exploration_prob)
    with open("./monitor_training/exploration_probability.pkl", "wb") as file:
        pickle.dump(self.exploration_Probabilities, file)

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

        event.BOMB_DROPPED: 5,
        event.BOMB_EXPLODED: 0,

        event.CRATE_DESTROYED: 5,
        event.COIN_FOUND: 10,
        event.COIN_COLLECTED: 100,

        event.KILLED_OPPONENT: 0,
        event.KILLED_SELF: -500,

        event.GOT_KILLED: -500,
        event.OPPONENT_ELIMINATED: 200,
        event.SURVIVED_ROUND: 50,

        # Custom events
        COIN_DIST_DECREASED: 5,
        STAYED_WITHIN_EXPLOSION_RADIUS: -5,
        MOVED_IN_SAFE_DIRECTION: 10,
        GOT_OUT_OF_EXPLOSION_RADIUS: 100,
        DROPPED_BOMB_WITH_NO_WAY_OUT: -100
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
    old_min_d = find_min_coin_distance(old_state["coins"], *old_state["self"][3])
    new_min_d = find_min_coin_distance(new_state["coins"], *new_state["self"][3])

    return new_min_d < old_min_d

def took_step_safe_direction(old_state, new_state):
    """
    Checks whether the agent moved away from a dangerous bomb.
    """
    bomb = find_closest_dangerous_bomb(old_state["bombs"], old_state["field"], old_state["self"][3])
    if bomb == None:
        return False
    old_dist = np.linalg.norm(np.array((old_state["self"][3][0], old_state["self"][3][1])) - np.array((bomb[0][0], bomb[0][1])))
    new_dist = np.linalg.norm(np.array((new_state["self"][3][0], new_state["self"][3][1])) - np.array((bomb[0][0], bomb[0][1])))

    return old_dist < new_dist

def stayed_within_explosion_radius(old_state, new_state):
    """
    Checks whether the agent continues to be in the radius of an explosion.
    """
    if not within_explosion_radius(old_state): return False
    else:
        if within_explosion_radius(new_state): return True
        else: return False

def got_out_of_explosion_radius(old_state, new_state):
    if not within_explosion_radius(old_state): return False
    else:
        if within_explosion_radius(new_state): return False
        else: return True

""" def reachable_safe_tile_exists(player_coords, field):
    x, y = player_coords[0], player_coords[1]

    print(x, y, field)
    for i in range(1, ARENA_WIDTH):
        if y+i >= ARENA_WIDTH: break
        if field[x+1][y+i] == 0:
            return True
        if field[x-1][y+i] == 0:
            return True
            
    for i in range(1, ARENA_WIDTH):
        if y-i >= ARENA_WIDTH: break
        if field[x+1][y-i] == 0:
            return True
        if field[x-1][y-i] == 0:
            return True
    
    for i in range(1, ARENA_LENGTH):
        if x+i >= ARENA_LENGTH: break
        if field[x+i][y+1] == 0:
            return True
        if field[x+i][y-1] == 0:
            return True
    
    for i in range(1, ARENA_LENGTH):
        if x-i >= ARENA_LENGTH: break
        if field[x-i][y+1] == 0:
            return True
        if field[x-i][y-1] == 0:
            return True
    
    return False """



def reachable_safe_tile_exists(player_coords, field, bombs):
    dangerous_bomb = find_closest_dangerous_bomb(bombs, field, player_coords)
    if dangerous_bomb == None:
        return None
    radius = get_blast_coords(dangerous_bomb[0], field)
    reachable_tiles = get_reachable_tiles(player_coords, field)
    
    for tile in reachable_tiles:
        if tile not in radius:
            return True
    else: return False


