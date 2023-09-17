import events as event
from typing import List
from collections import namedtuple, deque
import pickle
import numpy as np
from .definitions import *
from .callbacks import state_to_features
from .state_to_feature_helpers import *
from .customEventAppender import appendCustomEvents

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
	# Training parameters
    self.model.learning_rate = LEARNING_RATE
    self.model.discount_factor = DISCOUNT_FACTOR
    
	# Variables for the decay of the exploration probability
    self.model.exploration_prob = STARTING_EXPLORATION_PROBABILITY
    self.decay_active = EXPLORATION_DECAY_ACTIVE
    self.epsilon_decay = EPSILON_DECAY
    self.epsilon_decay_after_rounds = DECAY_AFTER_ROUNDS
    self.last_decayed_in_round = 0
    
	# For plotting
    self.model.total_reward = 0

    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=NUMBER_OF_RELEVANT_STATES)
    self.model.number_of_previous_states = NUMBER_OF_RELEVANT_STATES
    
	# Load the plot data back into memory if continue-training is true
    self.model.total_rewards = []
    self.model.total_qTable_size = []
    self.model.exploration_Probabilities = []
    
    if self.continue_training:
        with open("./monitor_training/total_rewards.pkl", "rb") as file:
            self.model.total_rewards = pickle.load(file)
        with open("./monitor_training/qTableSize.pkl", "rb") as file:
            self.model.total_qTable_size = pickle.load(file)
        with open("./monitor_training/exploration_probability.pkl", "rb") as file:
            self.model.exploration_Probabilities = pickle.load(file)
    
    hyperparameters = [self.model.learning_rate,
                       self.model.discount_factor,
                       self.decay_active,
                       self.epsilon_decay,
                       self.epsilon_decay_after_rounds,
                       self.model.number_of_previous_states]

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
    appendCustomEvents(self, events, new_game_state, old_game_state)

	# Safe the last n states in a dequeue
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(state_to_features(old_game_state),
                                       self_action,
                                       state_to_features(new_game_state),
                                       reward))

	# Only train if dequeue has n elemenets
    if len(self.transitions) == self.model.number_of_previous_states:
        self.model.train(self.transitions)
    
	# Calculate the total reward for this round
    self.model.total_reward += reward

	# Save last player position
    new_position = get_player_coordinates(new_game_state["self"])
    self.model.update_last_positions(new_position)

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
    self.transitions.append(Transition(state_to_features(last_game_state),
                                       last_action,
                                       None,
                                       reward_from_events(self, events)))

	# Train the remaining steps with reduced view into the future
	# Review this because it may be that you should leave this away
    for i in range(len(self.transitions), 1, -1):
        self.model.train(self.transitions)
        self.transitions.popleft()
        
	# Decay the exploration probability
    decay_exploration_prob(self, last_game_state["round"]) 

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

	# Store rewards
	# Add rewards of final steps, since there is no training
    self.model.total_reward += reward_from_events(self, events)
    self.model.total_rewards.append(self.model.total_reward)
    with open("./monitor_training/total_rewards.pkl", "wb") as file:
        pickle.dump(self.model.total_rewards, file)
    # Set total rewards to 0 for next round
    self.model.total_reward = 0
        
	# Store qTable size
    self.model.total_qTable_size.append(len(self.model.q_table))
    with open("./monitor_training/qTableSize.pkl", "wb") as file:
        pickle.dump(self.model.total_qTable_size, file)
        
	# Store exploration probability
    self.model.exploration_Probabilities.append(self.model.exploration_prob)
    with open("./monitor_training/exploration_probability.pkl", "wb") as file:
        pickle.dump(self.model.exploration_Probabilities, file)

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

        event.KILLED_OPPONENT: 0,
        event.KILLED_SELF: -600,

        event.GOT_KILLED: -300,
        event.OPPONENT_ELIMINATED: 200,
        event.SURVIVED_ROUND: 100,

        # Collect coins
        # COIN_DIST_DECREASED: 5,

        # Blow up Crates
        DROPPED_BOMB_NEAR_CRATE: 50,
        # GOT_OUT_OF_EXPLOSION_RADIUS: 20,
        # DROPPED_BOMB_WITH_NO_WAY_OUT: -100,
        SURVIVED_EXPLOSION: 5,
		# RUN_AWAY_FROM_BOMB_IF_ON_TOP: 25,
         
		 # Safty
        MOVED_IN_SAFE_DIRECTION: 15,
		MOVED_CLOSER_TO_SAVE_TILE: 10,
        MOVED_AWAY_FROM_SAVE_TILE: -10
    }
    
    total_reward = 0
    for instance in event_sequence:
        if instance in rewards:
            total_reward += rewards[instance]
            
    self.logger.info(f"Gained {total_reward} total reward for events {', '.join(event_sequence)}")
    return total_reward

def decay_exploration_prob(self, round_number):
        """
        Reduces the exploration probability for each training round gradually
        """
        
        if(not self.decay_active):
            return

		# Only true every epsilon_decay_after_rounds rounds
        if round_number == self.last_decayed_in_round + self.epsilon_decay_after_rounds:
            self.last_decayed_in_round = round_number
            self.model.exploration_prob *= self.epsilon_decay
    


