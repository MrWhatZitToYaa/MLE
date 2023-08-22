import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import definitions as d

class QLearning:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_table = {}
        
    def action_to_actionNum(action):
        """
        Converts action string to action number
        """
        return d.ACTIONS.index(action)
	
    def get_values_for_state(self, state):
        """
        Returns values of the actions in a given state if state exists
        Otherwise returns array of zeros
        """
        return self.q_table.get(state, [0.0] * d.NUM_OF_ACTIONS)
    
    def get_action_for_state(self, state):
        """
        Returns action for a given state
        """
        # probabilistic choice of action
		# TODO: code line aus anderem Github repo kopiert, funktion muss noch verstanden werden und bissel abändern oder löschen falls nicht verwedent!
    	#next_action = np.random.choice(ACTIONS, p=torch_functions.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    
        # choose action with maximum reward
        next_action = d.ACTIONS[np.argmax(self.model.get_values_for_state(state))]
        return next_action

    def update_q_table(self, old_state, action, reward, next_state):
        """
        Iterative approach to update the q-table
        TODO: Verify this is correct
        """
        best_next_action = self.model.get_action_for_state(next_state)
        best_next_actionNum = self.model.action_to_actionNum(best_next_action)
        actionNum = self.model.action_to_actionNum(action)
        
        old_q_value = self.model.get_values_for_state(old_state)[actionNum]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - old_q_value)
        
        self.q_table[next_state][action] = new_q_value

    def decay_exploration_prob(self):
        """
        Reduces the exploration probability for each traning round gradually
        TODO: ChatGPT proposed this, discuss usefullness and if we wanna do this
        """
        self.exploration_prob *= 0.95

    def train(self, old_state, action, reward, next_state):
        total_reward = 0
        
        action = self.model.get_action_for_state(old_state)
        self.update_q_table(old_state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
        
        #self.decay_exploration_prob()
        
        print(f"Total Reward: {total_reward}")

# Example usage
if __name__ == "__main__":
    num_states = 16
    num_actions = 4
    learning_rate = 0.1
    discount_factor = 0.99
    exploration_prob = 1.0

    q_learning_agent = QLearning(num_states, num_actions, learning_rate, discount_factor, exploration_prob)

    num_episodes = 1000
    max_steps_per_episode = 100
    env = YourEnvironment()

    q_learning_agent.train(num_episodes, max_steps_per_episode, env)