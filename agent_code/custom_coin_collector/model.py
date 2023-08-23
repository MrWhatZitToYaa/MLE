import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .definitions import *

class QLearning:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
	    
	self.total_reward = 0
	self.exploration_prob = 1
        self.q_table = {}
        
    def action_to_actionNum(self, action):
        """
        Converts action string to action number
        """
        return ACTIONS.index(action)
	
    def get_values_for_state(self, state):
        """
        Returns values of the actions in a given state if state exists
        Otherwise returns array of zeros
        """
        q_table = self.q_table.get(state, [0.0] * NUM_OF_ACTIONS)
        return q_table
    
    def get_action_for_state(self, state):
        """
        Returns action for a given state
        """
        # probabilistic choice of action
		# TODO: code line aus anderem Github repo kopiert, funktion muss noch verstanden werden und bissel abändern oder löschen falls nicht verwedent!
    	#next_action = np.random.choice(ACTIONS, p=torch_functions.softmax(self.model.forward(game_state), dim=0).detach().numpy())

        # choose action with maximum reward
        next_action = ACTIONS[np.argmax(self.get_values_for_state(state))]
        return next_action

    def update_q_table(self, old_state, action, reward, next_state):
        """
        Iterative approach to update the q-table
        TODO: Verify this is correct AND CHECK WHY THE FIRST COMMENTED OUT VERSION IS NOT WORKING EVEN THOUGH SAME CODE
        """
        """
        best_next_action = self.get_action_for_state(next_state)
        best_next_actionNum = self.action_to_actionNum(best_next_action)
        qValue_best_next_action = self.get_values_for_state(next_state)
        actionNum = self.action_to_actionNum(action)
        
        old_q_value = self.get_values_for_state(old_state)[actionNum]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_actionNum] - old_q_value)
        
        self.q_table[old_state][actionNum] = new_q_value
        """
        qValue_best_next_action = self.q_table.get(next_state, [0.0] * NUM_OF_ACTIONS)
        best_next_action = ACTIONS[np.argmax(qValue_best_next_action)]
        best_next_actionNum = self.action_to_actionNum(best_next_action)
        actionNum = self.action_to_actionNum(action)
        
        old_q_value = self.q_table.get(old_state, [0.0] * NUM_OF_ACTIONS)[actionNum]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * qValue_best_next_action[best_next_actionNum] - old_q_value)
        
        if old_state not in self.q_table:
            self.q_table[old_state] = [0.0] * self.num_actions
        self.q_table[old_state][actionNum] = new_q_value

    def decay_exploration_prob(self):
        """
        Reduces the exploration probability for each traning round gradually
        TODO: ChatGPT proposed this, discuss usefullness and if we wanna do this
        """
        self.exploration_prob *= EPSILON_DECAY

    def train(self, old_state, action, reward, next_state):    
        action = self.get_action_for_state(old_state)
        self.update_q_table(old_state, action, reward, next_state)
        
        self.total_reward += reward
        
        self.decay_exploration_prob()
        
