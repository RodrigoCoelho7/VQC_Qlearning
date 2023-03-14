import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tensorflow_quantum as tfq
from vqc.vqc_circuit import VQCCIRCUIT
import numpy as np
import cirq


class MyPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.

    Arguments: 
        num_qubits: number of qubits in the circuit
        n_layers: number of layers in the circuit
        observables: list of observables to be measured
        circuit_arch: architecture of the circuit
        data_reuploading: version of data reuploading to be used
        activation: activation function to be applied to the lmbd[i][j]s_i
        name: name of the VQC
    """

    def __init__(self, num_qubits, n_layers, observables, circuit_arch, data_reuploading,measurement ,activation="linear", name="MyPQC"):
        super(MyPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.data_reuploading = data_reuploading

        vqc = VQCCIRCUIT(num_qubits,n_layers,circuit_arch,data_reuploading, measurement)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        
        if self.data_reuploading == "basic" or self.data_reuploading == "schuld":
            lmbd_init = tf.ones(shape=(len(input_symbols),))
            self.lmbd = tf.Variable(
                initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
            )
        else:
            lmbd_init = tf.ones(shape=(len(input_symbols),))
            self.lmbd = tf.Variable(
                initial_value=lmbd_init, dtype="float32", trainable= False, name="lambdas",
            )
        
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
        if self.data_reuploading == "basic" or self.data_reuploading == "schuld":
            tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        else:
            tiled_up_inputs = tf.tile(inputs[0], multiples=[1, 1])

        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)

        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output


class UniversalQuantumClassifier(tf.keras.layers.Layer):
    
    def __init__(self,num_qubits,num_layers,observables,circuit_arch, data_reuploading, measurement, activation = "linear", name = "UQC"):
        super(UniversalQuantumClassifier, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.data_reuploading = data_reuploading
        self.circuit_arch = circuit_arch

        vqc = VQCCIRCUIT(num_qubits, num_layers, circuit_arch, data_reuploading, measurement)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        w_init = tf.random_normal_initializer(mean=0.0, stddev=1)
        self.w = tf.Variable(
            initial_value = w_init(shape = (self.num_layers,4), dtype = "float32"),
            trainable = True, name = "w")
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value = b_init(shape = (self.num_layers,), dtype = "float32"),
            trainable = True, name = "b")
        
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # Batch dim gives the dimension of the batch (16,32,etc)
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        #tiled_up_circuits tiles the required number of circuits for the batch size
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        #tiled_up_thetas tiles the required number of thetas for the batch size
        tiled_up_thetas = tf.tile(tf.multiply(self.theta,2), multiples=[batch_dim, 1])
        
        #Here I dont need to tile the inputs. The inputs[0] tensor has shape (batch_size, state_size).
        #Multiplying it by the transpose of the weights which has shape (state_size, num_layers) will yield
        #a tensor of shape (batch_size, num_layers) which is what I want.
        inputs_times_weights = tf.matmul(inputs[0], tf.multiply(self.w,2), transpose_b=True)

        #Now I need to add the bias. The bias has shape (num_layers,) so I need to tile it to have shape (batch_size, num_layers)
        tiled_up_b = tf.reshape(tf.tile(self.b, multiples = [batch_dim]), (batch_dim, self.num_layers))
        inputs_times_w_plus_b = inputs_times_weights + tf.multiply(tiled_up_b,2)

        # The activation function is applied to the inputs_times_w_plus_b tensor
        activated_inputs = tf.keras.layers.Activation(self.activation)(inputs_times_w_plus_b)

        # The thetas and the activated inputs are concatenated and then the indices are gathered
        joined_vars = tf.concat([tiled_up_thetas, activated_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        # The computation layer is applied to the circuits and the variables
        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output
        


        


