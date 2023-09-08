import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cirq
from collections import deque
from vqc.vqc_circuits import UQC
from model.output_scaling import LocalSkolikRescaling
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max
from wrappers import ContinuousEncoding, NothingEncoding

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
model_quantum = False
state_dim = 6
num_actions = 3
activation = "relu"

# Prepare the optimizers
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

# Parameters for the training
gamma = 0.99
num_episodes = 500
max_memory_length = 500000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
policy = EGreedyExpStrategy(epsilon, epsilon_min, decay_epsilon)
batch_size = 64
steps_per_update = 1 # Train the model every x steps
steps_per_target_update = 1000 # Update the target model every x steps
operator = Max()
parameters_relative_change = True
entanglement_study = False

#Choose the environment
environment = "Acrobot-v1"
input_encoding = NothingEncoding
early_stopping = False
acceptance_reward = 195
necessary_episodes = 100



#He_normal initialization
#relu for the first three layers, linear for the last one

#Best performance so far