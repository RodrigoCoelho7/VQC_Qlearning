import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cirq
from collections import deque
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max
from vqc.vqc_circuits import SkolikSchuld
from model.output_scaling import LocalSkolikRescaling
from wrappers import ContinuousEncoding, NothingEncoding

"""
This cript is to test whether importing the input encoding wrapper in the config file
instead of the DQN file is what changed the code for the worse.
"""

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Classical Model
model_quantum = False
state_dim = 4
num_actions = 2
activation = "relu"

# Training the classical model
learning_rate_var = 0.001
optimizer_in =  None
optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
optimizer_out = None
optimizer_bias = None

# Assign the model parameters to each optimizer
w_in = None
w_var = 0
w_out = None
w_bias = None

# Parameters for the training
gamma = 0.99
num_episodes = 1000
max_memory_length = 10000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
policy = EGreedyExpStrategy(epsilon, epsilon_min, decay_epsilon)
batch_size = 16
steps_per_update = 1 # Train the model every x steps
steps_per_target_update = 1 # Update the target model every x steps
operator = Max()
parameters_relative_change = True
entanglement_study = True

#Choose the environment
environment = "CartPole-v0"
input_encoding = NothingEncoding
early_stopping = False
acceptance_reward = 195
necessary_episodes = 100