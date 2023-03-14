import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cirq
from collections import deque

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
num_qubits = 1
num_layers = 5
num_actions = 2
circuit_arch = "uqc"
data_reuploading = "baseline"
qubits = cirq.GridQubit.rect(1, num_qubits)
ops = cirq.Z(qubits[0])
observables = [-ops, ops]
measurement = "lastZ"
rescaling_type = "localexpectation"
state_dim = 4

# Parameters for the training
gamma = 0.99
num_episodes = 3000
max_memory_length = 10000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
batch_size = 16
steps_per_update = 10 # Train the model every x steps
steps_per_target_update = 30 # Update the target model every x steps

# Prepare the optimizers
learning_rate_in = 0.001
learning_rate_var = 0.001
learning_rate_out = 0.1
optimizer_in =  tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

# Assign the model parameters to each optimizer
w_in = 1
w_var = 0
w_out = 2

#Choose the environment
environment = "CartPole-v0"
input_encoding = "continuous"
early_stopping = False
acceptance_reward = 195
necessary_episodes = 25