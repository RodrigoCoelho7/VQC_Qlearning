import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

"""
Defines whether the agent is always going to consider the maximum Q-values of the successor states or some other soother operation.
"""

class Max():
    def apply(self, x):
        return tf.reduce_max(x, axis=1)

class MellowMax():
    def __init__(self, omega, num_actions):
        self.omega = omega
        self.num_actions = num_actions
    
    def apply(self,x):
        exp_x = tf.exp(self.omega * x)
        sum_exp_x = tf.reduce_sum(exp_x, axis=1)
        return tf.math.log((1/self.num_actions) * sum_exp_x) / self.omega