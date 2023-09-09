import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np

"""
Implements the policies for the agents:

EGreedyExpStrategy: Implements the exponentially decaying epsilon-greedy policy
"""

class EGreedyExpStrategy():
    def __init__(self, epsilon, epsilon_min, decay_epsilon):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_epsilon = decay_epsilon
    
    def select_action(self, state, model, n_actions):
        state_array = np.array(state) 
        state = tf.convert_to_tensor([state_array])
        coin = np.random.random()
        if coin > self.epsilon:
            q_vals = model([state])
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(n_actions)
        return action
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_epsilon)