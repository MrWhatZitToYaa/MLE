import numpy as np
from .state_to_feature_helper import state_to_features
import time
import os
import pickle


class CustomModel:
    modelDict = {(0, 0, 0, 0): [0, 0, 0, 0, 0, 0]}
    qDictFileName = str

    lastPositions = [(np.NINF, np.NINF), (np.inf, np.inf)]

    def __init__(self):
        print('new model created')
        self.actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB'])
        self.action_probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])

        self.learning_rate = 0.01
        self.discount_factor = 0.99

        self.qDictFileName = "custom_model.pt"

        if not os.path.isfile(self.qDictFileName):
            print('created new dict file')
            pickle.dump(self.modelDict, open(self.qDictFileName, "wb"))

    def predict_action(self, game_state):
        features = state_to_features(game_state)
        qTable = pickle.load(open(self.qDictFileName, "rb"))

        if features in qTable:
            rewards = qTable[features]
            action = np.argmax(rewards)
            print('action according to best feature in Q Table')
        else:
            action = np.random.choice(len(self.actions) - 1)
            print('random action')
        return action

    def update_last_positions(self, position: tuple):
        self.lastPositions[0] = self.lastPositions[1]
        self.lastPositions[1] = position

    def state_exists_in_table(self, state, table):
        return next((True for elem in table if np.array_equal(elem, state)), False)

    def update_qtable(self, old_state, next_state, action_taken, total_reward, end_of_game):
        currentQTable = pickle.load(open(self.qDictFileName, "rb"))

        actionIndex = np.where(self.actions == action_taken)[0]
        #if end_of_game:
        #    print(next_state)
        old_state_features = state_to_features(old_state)

        next_state_features = None
        if next_state is not None:
            next_state_features = state_to_features(next_state)

        if old_state_features in currentQTable:
            old_state_rewards = currentQTable[old_state_features]
            old_reward = old_state_rewards[actionIndex]
            if next_state_features in currentQTable and next_state_features is not None:
                next_state_rewards = currentQTable[next_state_features]
                max_value_of_next_state = np.max(next_state_rewards)
            else:
                max_value_of_next_state = 0

            '''
            new_value = (1 - self.alpha) * old_reward + self.alpha * (
                        total_reward + self.gamma * max_value_of_next_state)
            '''
            if end_of_game:
                new_q_value = self.learning_rate * old_reward + self.discount_factor * total_reward
            else:
                new_q_value = old_reward + self.learning_rate * (total_reward + self.discount_factor * max_value_of_next_state) - old_reward

            currentQTable[old_state_features][actionIndex] = new_q_value

        else:
            # old feature not yet in current table
            new_row = np.zeros([len(self.actions)])
            new_row[actionIndex] = total_reward
            currentQTable.update({old_state_features: new_row})

        pickle.dump(currentQTable, open(self.qDictFileName, "wb"))

