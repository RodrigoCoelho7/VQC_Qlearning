import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import cirq
import pickle
import matplotlib.pyplot as plt
from model.q_learning_agent import QLearningAgent
from DQN.dqn import DQN
from collections import deque
import sys
import time

# To test the local vs global measurements, i changed the ops and observables hyperparameters
#as well as the QLearnignAgent and the VQCCircuit

#sys.argv[1] -> Data Reuploading Type (v0, v1 or v2)
#sys.argv[2] -> Number of the agent
#sys.argv[3] -> Input encoding technique (sc,scv2,c)
#sys.argv[4] -> Path to store pickle file
#sys.argv[5] -> outscalingmethod (localskolik,localnormal,globalskolik,globalnormal)
#sys.argv[6] -> measurement type (global or local)

#I tested localskolik global


if __name__ == "__main__":
    start_time = time.time()

    # Parameters for the VQC
    num_qubits = 1
    num_layers = 5
    num_actions = 2
    circuit_arch = "uqc"
    data_reuploading = sys.argv[1]
    qubits = cirq.GridQubit.rect(1, num_qubits)
    #ops = [cirq.Z(q) for q in qubits]
    #observables = [ops[0]*ops[1], ops[2]*ops[3]]tt
    ops = cirq.Z(qubits[-1])
    observables = [-ops,+ops]
    measurement = sys.argv[6]
    rescaling_type = sys.argv[5]

    # Create the Models
    model = QLearningAgent(num_qubits, num_layers,observables, circuit_arch, data_reuploading, False,measurement, rescaling_type)
    model_target = QLearningAgent(num_qubits, num_layers,observables, circuit_arch, data_reuploading, True, measurement, rescaling_type)
    model_target.set_weights(model.get_weights())
    
    # Parameters for the training
    gamma = 0.99
    num_episodes = 3000
    max_memory_length = 10000 # Maximum replay length
    replay_memory = deque(maxlen=max_memory_length)
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
    batch_size = 16
    steps_per_update = 1 # Train the model every x steps
    steps_per_target_update = 1 # Update the target model every x steps

    # Prepare the optimizers

    optimizer_in = None if sys.argv[1] == "v0" else tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in = None if sys.argv[1] == "v0" else 1
    w_var = 0
    w_out = 1 if sys.argv[1] == "v0" else 2

    #Choose the environment
    environment = "CartPole-v0"
    input_encoding = sys.argv[3]
    early_stopping = False
    acceptance_reward = 195
    necessary_episodes = 25

    # Create the agent
    agent = DQN(model, model_target, gamma, num_episodes, max_memory_length,
                replay_memory, epsilon, epsilon_min, decay_epsilon, batch_size,
                steps_per_update, steps_per_target_update, optimizer_in, optimizer_out, optimizer_var,
                w_in, w_var, w_out, input_encoding, early_stopping)
    
    agent.train(environment, num_actions, acceptance_reward, necessary_episodes)

    path = sys.argv[4]
    filename = f"CartPolev0_uqc_{sys.argv[1]}_ls_optimal_hyper_globalnormal_outscaling_{sys.argv[2]}.pkl"
    agent.store_pickle(path, filename)

    print("--- %s seconds ---" % (time.time() - start_time))