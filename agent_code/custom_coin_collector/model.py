import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import definitions as d
# Note ChatGPT code. Muss angepasst werden für unseren Verwendungsfall aber ich bin jetzt zu müde also lieber morgen

class QLearning:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, exploration_prob):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

    def decay_exploration_prob(self):
        self.exploration_prob *= 0.95

    def train(self, num_episodes, max_steps_per_episode, env):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update_q_table(state, action, reward, next_state)

                total_reward += reward
                state = next_state

                if done:
                    break

            self.decay_exploration_prob()

            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

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