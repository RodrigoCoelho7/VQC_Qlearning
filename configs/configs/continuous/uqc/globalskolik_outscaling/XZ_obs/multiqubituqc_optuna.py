import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max
from model.output_scaling import GlobalSkolikRescaling
import tensorflow as tf
from collections import deque

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
num_actions = 2
rescaling_type = GlobalSkolikRescaling
state_dim = 4

# Parameters for the training
gamma = 0.99
steps_per_update = 1
steps_per_target_update = 1
batch_size = 48
max_memory_length = 70000
replay_memory = deque(maxlen=max_memory_length)
num_episodes = 250
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
policy = EGreedyExpStrategy(epsilon, epsilon_min, decay_epsilon)
operator = Max()
activation = "linear"

# Assign the model parameters to each optimizer
learning_rate_in = 0.001
learning_rate_out = 0.1
learning_rate_var = 0.001
optimizer_in = tf.keras.optimizers.Adam(learning_rate=learning_rate_in, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rate_out, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
optimizer_bias = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
w_in = 1
w_var = 0
w_bias = 2
w_out = 3

#Choose the environment
environment = "CartPole-v0"
input_encoding = "continuous"
early_stopping = False
acceptance_reward = 195
necessary_episodes = 25