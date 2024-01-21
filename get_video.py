import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import gym
from vqc.vqc_circuits import SkolikSchuld
import cirq
import tensorflow_quantum as tfq
from data_analysis.analysis_functions import Analysis
import imageio

# We need to define a model that will already receive the weights

class DataReupPQC(tf.keras.layers.Layer):

    def __init__(self, vqc, observables, activation, weights, name="MyPQC"):
        super(DataReupPQC, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs
        self.n_layers = vqc.num_layers

        self.rotation_weights = weights[0]
        self.input_weights = weights[1]
        self.output_weights = weights[2]

        self.theta = tf.Variable(initial_value=self.rotation_weights, trainable=False, name="thetas")
        
        self.lmbd = tf.Variable(initial_value=self.input_weights, trainable=False, name="lambdas")
        
        # Define explicit symbol order,
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        #Inputs is a list of tensors, the first one is the input data with shape (batch_size, state_dim)
        # Batch dim gives the dimension of the batch (16,32,etc)
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)

        #tiled_up_circuits tiles the required number of circuits for the batch size
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)

        #tiled_up_thetas tiles the required number of thetas for the batch size
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])

        #tiled_up_inputs tiles the required number of inputs (states) for the number of layers in the case of data reup
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])

        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)

        squashed_inputs = tf.atan(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output
    

class LocalSkolikRescaling(tf.keras.layers.Layer):
    def __init__(self, weights):
        super(LocalSkolikRescaling, self).__init__()
        self.w = tf.Variable(
            initial_value=weights[2], dtype="float32",
            trainable=False, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))
    
def QuantumQLearningAgent(vqc, quantum_model, observables, target,state_dim, rescaling_type, activation,weights):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')
    pqc = quantum_model(vqc, observables, activation, weights)([input_tensor])
    process = tf.keras.Sequential([rescaling_type(weights)], name=target*"Target"+"Q-values")
    Q_values = process(pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model

if __name__ == "__main__":
    
    # Parameters for the VQC
    model_quantum = True
    num_qubits = 4
    num_layers = 5
    vqc = SkolikSchuld(num_qubits, num_layers)
    qubits = cirq.GridQubit.rect(1, num_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [ops[0]*ops[1], ops[2]*ops[3]]
    rescaling_type = LocalSkolikRescaling
    state_dim = 4
    activation = "linear"

    path = "../results/thesis/1.1/skolik_datareup"
    skolik_datareup = Analysis(path)
    weights = skolik_datareup.get_final_weights()[2]

    quantum_model = DataReupPQC
    model = QuantumQLearningAgent(vqc, quantum_model, observables, False, state_dim, rescaling_type, activation, weights)

    def select_action(state, model):
        state_array = np.array(state) 
        state = tf.convert_to_tensor([state_array])
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
        return action
    
    
    env = gym.make('CartPole-v0')
    frames = []
    
    state = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode = 'rgb_array'))
        action = select_action(state, model)
        state, reward, done, info = env.step(action)
    env.close()

    np.save("cartpole_frames.npy", frames)

