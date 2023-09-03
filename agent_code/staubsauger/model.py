import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

from .definitions import *

class QLearning:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        
        # Q-table related
        self.q_table = {}
        
		# ?
        self.lastPositions = [(np.NINF, np.NINF), (np.inf,np.inf), (np.inf, np.inf)]
        
    def action_to_actionNum(self, action):
        """
        Converts action string to action number
        """
        return ACTIONS.index(action)
	
    def get_qValues_for_state(self, state):
        """
        Returns values of the actions in a given state if state exists
        Otherwise returns array of zeros
        """
        q_table_row = self.q_table.get(state, [0.0] * self.num_actions)
        return q_table_row
    
    def get_action_for_state(self, state):
        """
        Returns action for a given state
        """

        # choose action with maximum reward or random action if all qValues are the same
        qValues = self.get_qValues_for_state(state)
        if all(element == qValues[0] for element in qValues):
            next_action = np.random.choice(ACTIONS, p=PROBABILITIES_FOR_ACTIONS)
        else:
            next_action = ACTIONS[np.argmax(qValues)]
        	# probabilistic choice of action
    		#next_action = np.random.choice(ACTIONS, self.get_qValues_for_state(state).numpy())
            
        return next_action
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Iterative approach to update the q-table
        """
        best_next_action = self.get_action_for_state(next_state)
        best_next_actionNum = self.action_to_actionNum(best_next_action)
        qValue_best_next_action = self.get_qValues_for_state(next_state)[best_next_actionNum]
        
        actionNum = self.action_to_actionNum(action)
        
        old_q_value = self.get_qValues_for_state(state)[actionNum]
        new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * qValue_best_next_action)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions

        self.q_table[state][actionNum] = new_q_value

    def decay_exploration_prob(self, round_number):
        """
        Reduces the exploration probability for each training round gradually
        """
        
        if(not self.decay_active):
            return

		# Only true if round number changes
        if round_number != self.last_round:
            self.last_round = round_number
            # Only true every epsilon_decay_after_rounds rounds
            if round_number == self.last_decayed_in_round + self.epsilon_decay_after_rounds:
                self.last_decayed_in_round = round_number
                self.exploration_prob *= EPSILON_DECAY

    def train(self, transitions, round_number):
        self.update_q_table(transitions)
        
		#decay exploration probability
        self.decay_exploration_prob(round_number)
        
    def update_last_positions(self, position: tuple):
        self.lastPositions[0] = self.lastPositions[1]
        self.lastPositions[1] = position