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
        
    def update_q_table_constrained(self, transitions, number_of_states_in_future):
        """
        Iterative approach to update the q-table
        Using SARSA n-step method
        :param transitions: Dequeue of transitions, each entry contains a named tuple of the form (s, a, r, s')
							The oldest tuple is stored in the beginning of the dequeue
        :param number_of_states_in_future: Number of states into the future that are used to update current q
										   value
        """
        
		# Keep in mind that we don't update the q value for the last action but for the action that was taken
		# NUMBER_OF_RELEVANT_STATES bofore the current one

		# Calculate previous q value
        state = transitions[0][0]
        action = transitions[0][1]

        actionNum = self.action_to_actionNum(action)
        previous_q_value = self.get_qValues_for_state(state)[actionNum]

		# Calculate sum of discounted rewards for next NUMBER_OF_RELEVANT_STATES actions
        sum_of_discounted_rewards = 0
        for i in range(number_of_states_in_future):
            sum_of_discounted_rewards += pow(self.discount_factor, i) * transitions[i][3]

		# Approximation for reward of rest of the game (Reward for best next action)
        rest = 0

        last_state = transitions[-1][2]
        best_next_action = self.get_action_for_state(last_state)
        best_next_actionNum = self.action_to_actionNum(best_next_action)
        qValue_best_next_action = self.get_qValues_for_state(last_state)[best_next_actionNum]

        rest = qValue_best_next_action * pow(self.discount_factor, number_of_states_in_future) 
        
        new_q_value = (1 - self.learning_rate) * previous_q_value + self.learning_rate * (sum_of_discounted_rewards + rest)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions

        self.q_table[state][actionNum] = new_q_value
    

    def train(self, transitions):
        self.update_q_table_constrained(transitions, len(transitions))       

    def update_last_positions(self, position: tuple):
        self.lastPositions[0] = self.lastPositions[1]
        self.lastPositions[1] = position