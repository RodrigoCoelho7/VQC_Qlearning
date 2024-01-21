import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cirq
from collections import deque
from vqc.vqc_circuits import UQC
from model.output_scaling import LocalExpectationRescaling
from DQN.policies import EGreedyExpStrategy
from DQN.operators import Max
from wrappers import AcrobotEncodingV2
from vqc.data_reup_model import FullEncodingMultiQubitUniversalQuantumClassifier

#circuit_arch = "skolik", "lock" or "uqc"
#data_reuploading = "baseline", "basic" or "schuld"
#measurement = "ZZ" or "lastZ"
#rescaling_type = "localskolik", "localexpectation", "globalskolik" or "globalexpectation"
#input_encoding = "scaled_continuous" or "continuous"

# Parameters for the VQC
model_quantum = True
num_qubits = 8
num_layers = 5
num_actions = 3
vqc = UQC(num_qubits, num_layers)
qubits = cirq.GridQubit.rect(1, num_qubits)
ops = [cirq.Z(qubits[0]), cirq.Z(qubits[1]), cirq.Z(qubits[2]), cirq.Z(qubits[3]), cirq.Z(qubits[4]), cirq.Z(qubits[5]), cirq.Z(qubits[6]), cirq.Z(qubits[7])]
observables = [ops[0]* ops[1], ops[2]* ops[3] *  ops[4]* ops[5], ops[6]* ops[7]]
rescaling_type = LocalExpectationRescaling
state_dim = 4
quantum_model = FullEncodingMultiQubitUniversalQuantumClassifier
activation = "linear"

# Parameters for the training
gamma = 0.99
steps_per_update = 5
batch_size = 32
steps_per_target_update = 250
max_memory_length = 50000
replay_memory = deque(maxlen=max_memory_length)
num_episodes = 500
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
policy = EGreedyExpStrategy(epsilon, epsilon_min, decay_epsilon)
operator = Max()
activation = "linear"
parameters_relative_change = False
entanglement_study = False


# Prepare the optimizers
learning_rate_in = 0.001
learning_rate_var = 0.001
learning_rate_out = 0.1
optimizer_in =  tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_bias = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

# Assign the model parameters to each optimizer
w_in = 1
w_var = 0
w_bias = 2
w_out = 3

#Choose the environment
environment = "Acrobot-v1"
input_encoding = AcrobotEncodingV2
early_stopping = False
acceptance_reward = -100
necessary_episodes = 100