import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cirq
from collections import deque
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max, MellowMax

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
num_qubits = 4
num_layers = 5
num_actions = 2
circuit_arch = "skolik"
data_reuploading = "schuld"
qubits = cirq.GridQubit.rect(1, num_qubits)
ops = [cirq.Z(q) for q in qubits]
observables = [ops[0]*ops[1], ops[2]*ops[3]]
measurement = "ZZ"
rescaling_type = "localskolik"
state_dim = 4

# Parameters for the training
gamma = 0.99
num_episodes = 150
max_memory_length = 10000 # Maximum replay length
replay_memory = deque(maxlen=max_memory_length)
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
policy = EGreedyExpStrategy(epsilon, epsilon_min, decay_epsilon)
batch_size = 16
steps_per_update = 1 # Train the model every x steps
steps_per_target_update = 1 # Update the target model every x steps
omega =  15
operator = MellowMax(omega, num_actions)

# Prepare the optimizers
learning_rate_in = 0.001
learning_rate_var = 0.001
learning_rate_out = 0.1
optimizer_in =  tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
optimizer_bias = None

# Assign the model parameters to each optimizer
w_in = 1
w_var = 0
w_out = 2
w_bias = None

#Choose the environment
environment = "CartPole-v0"
input_encoding = "continuous"
early_stopping = False
acceptance_reward = 195
necessary_episodes = 25