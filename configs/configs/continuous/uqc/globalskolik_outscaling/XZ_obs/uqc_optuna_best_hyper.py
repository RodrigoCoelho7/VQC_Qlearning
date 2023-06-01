import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cirq
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max
from vqc.vqc_circuits import UQC
from model.output_scaling import LocalExpectationRescaling
import tensorflow as tf
from collections import deque
from wrappers import ContinuousEncoding
from vqc.data_reup_model import UniversalQuantumClassifier

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
num_actions = 2
num_qubits = 1
num_layers = 5
vqc = UQC(num_qubits, num_layers)
qubits = cirq.GridQubit.rect(1, num_qubits)
ops = [cirq.Z(qubits[0]), cirq.X(qubits[0])]
observables = [ops[0], ops[1]]
rescaling_type = LocalExpectationRescaling
state_dim = 4
pqc = UniversalQuantumClassifier

# Parameters for the training
gamma = 0.99
batch_size = 48
steps_per_update = 1
steps_per_target_update = 3
max_memory_length = 50000
replay_memory = deque(maxlen=max_memory_length)
num_episodes = 3000
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
input_encoding = ContinuousEncoding
early_stopping = False
acceptance_reward = 195
necessary_episodes = 25