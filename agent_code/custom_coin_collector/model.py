import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .definitions import *

class QLearning:
    def __init__(self, num_actions, learning_rate, discount_factor, startin_exploration_probability):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.total_reward = 0
        self.exploration_prob = startin_exploration_probability
        self.q_table = {}
        
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
        # probabilistic choice of action
    	#next_action = np.random.choice(ACTIONS, self.get_qValues_for_state(state).numpy())

        # choose action with maximum reward
        next_action = ACTIONS[np.argmax(self.get_qValues_for_state(state))]
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

    def decay_exploration_prob(self):
        """
        Reduces the exploration probability for each traning round gradually
        TODO: ChatGPT proposed this, discuss usefullness and if we wanna do this
        """
        self.exploration_prob *= EPSILON_DECAY

    def train(self, state, action, reward, next_state):
        self.update_q_table(state, action, reward, next_state)
        self.total_reward += reward
        #self.decay_exploration_prob()
        
