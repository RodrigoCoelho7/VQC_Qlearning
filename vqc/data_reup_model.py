import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq

class BaselinePQC(tf.keras.layers.Layer):
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

    def __init__(self, vqc,state_dim, observables, activation="linear", name="MyPQC"):
        super(BaselinePQC, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        
        lmbd_init = tf.ones(shape=(len(input_symbols),))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable= True, name="lambdas",
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
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, 1])

        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)

        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output

class DataReupPQC(tf.keras.layers.Layer):
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

    def __init__(self, vqc,state_dim, observables, activation, name="MyPQC"):
        super(DataReupPQC, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs
        self.n_layers = vqc.num_layers

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        
        lmbd_init = tf.ones(shape=(len(input_symbols),))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
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
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])

        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)

        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output


class UniversalQuantumClassifier(tf.keras.layers.Layer):
    
    def __init__(self, vqc,state_dim,observables, activation, name = "UQC"):
        super(UniversalQuantumClassifier, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs
        self.n_layers = vqc.num_layers
        self.state_dim = state_dim

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        w_init = tf.random_normal_initializer(mean=0.0, stddev=1)
        self.w = tf.Variable(
            initial_value = w_init(shape = (self.n_layers,self.state_dim), dtype = "float32"),
            trainable = True, name = "w")
        
        b_init = tf.zeros_initializer()
        #b_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        self.b = tf.Variable(
            initial_value = b_init(shape = (self.n_layers,), dtype = "float32"),
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
        tiled_up_b = tf.reshape(tf.tile(self.b, multiples = [batch_dim]), (batch_dim, self.n_layers))
        inputs_times_w_plus_b = inputs_times_weights + tf.multiply(tiled_up_b,2)

        # The activation function is applied to the inputs_times_w_plus_b tensor
        activated_inputs = tf.keras.layers.Activation(self.activation)(inputs_times_w_plus_b)

        # The thetas and the activated inputs are concatenated and then the indices are gathered
        joined_vars = tf.concat([tiled_up_thetas, activated_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        # The computation layer is applied to the circuits and the variables
        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output
        
class MultiQubitUniversalQuantumClassifier(tf.keras.layers.Layer):
    
    def __init__(self, vqc, state_size ,observables, activation, name = "UQC"):
        super(MultiQubitUniversalQuantumClassifier, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs
        self.num_layers = vqc.num_layers
        self.num_qubits = vqc.num_qubits
        self.state_size = state_size

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.w = tf.Variable(
            initial_value = w_init(shape = (self.num_layers, self.num_qubits,self.state_size//self.num_qubits), dtype = "float32"),
            trainable = True, name = "w")
        
        #b_init = tf.zeros_initializer()
        b_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.b = tf.Variable(
            initial_value = b_init(shape = (self.num_layers, self.num_qubits), dtype = "float32"),
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
        
        #The inputs[0] tensor has shape (batch_size, state_size). I need it to have shape (batch_dim,num_layers,
        # num_qubits, state_size//num_qubits), so I first need to reshape it, then tile it and then reshape it.
        reshaped_inputs = tf.reshape(inputs[0], (batch_dim, self.num_qubits, self.state_size//self.num_qubits))
        tiled_inputs = tf.tile(reshaped_inputs, multiples = [1, self.num_layers, 1])
        reshaped_tiled_inputs = tf.reshape(tiled_inputs, (batch_dim, self.num_layers, self.num_qubits, self.state_size//self.num_qubits))
        #Now, we can simply do the element-wise cross product between the weights and the inputs and the
        #resulting tensor will have shape (batch_dim, num_layers, num_qubits)
        inputs_times_weights = tf.reduce_sum(tf.multiply(self.w,2) * reshaped_tiled_inputs, axis = -1, keepdims = False)

        #Now I need to add the bias. The bias has shape (num_layers,num_qubits) so I need to tile it to have shape 
        # (batch_size, num_layers, num_qubits)
        tiled_up_bias = tf.reshape(tf.tile(self.b, multiples = [batch_dim, 1]), (batch_dim, self.num_layers, self.num_qubits))
        inputs_times_weights_plus_b = inputs_times_weights + tf.multiply(tiled_up_bias,2)
        reshaped_inputs_times_weights_plus_b = tf.reshape(inputs_times_weights_plus_b, (batch_dim, self.num_layers * self.num_qubits))

        # The activation function is applied to the inputs_times_w_plus_b tensor
        activated_inputs = tf.keras.layers.Activation(self.activation)(reshaped_inputs_times_weights_plus_b)

        # The thetas and the activated inputs are concatenated and then the indices are gathered
        joined_vars = tf.concat([tiled_up_thetas, activated_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        # The computation layer is applied to the circuits and the variables
        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output
    

class FullEncodingMultiQubitUniversalQuantumClassifier(tf.keras.layers.Layer):
    
    def __init__(self, vqc, state_size ,observables, activation, name = "UQC"):
        super(FullEncodingMultiQubitUniversalQuantumClassifier, self).__init__(name=name)

        circuit, theta_symbols, input_symbols = vqc.circuit, vqc.parameters, vqc.inputs
        self.num_layers = vqc.num_layers
        self.num_qubits = vqc.num_qubits
        self.state_size = state_size

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        w_init = tf.random_normal_initializer(mean=0.0, stddev=1)
        self.w = tf.Variable(
            initial_value = w_init(shape = (self.num_layers, self.num_qubits,self.state_size), dtype = "float32"),
            trainable = True, name = "w")
        
        b_init = tf.zeros_initializer()
        #b_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        self.b = tf.Variable(
            initial_value = b_init(shape = (self.num_layers, self.num_qubits), dtype = "float32"),
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
        
        #The inputs[0] tensor has shape (batch_size, state_size). I need it to have shape (batch_dim,num_layers,
        # num_qubits, state_size//num_qubits), so I first need to reshape it, then tile it and then reshape it.
        reshaped_inputs = tf.reshape(inputs[0], (batch_dim, 1, 1, self.state_size))
        tiled_inputs = tf.tile(reshaped_inputs, multiples = [1, self.num_layers,self.num_qubits, 1])
        #Now, we can simply do the element-wise cross product between the weights and the inputs and the
        #resulting tensor will have shape (batch_dim, num_layers, num_qubits)
        inputs_times_weights = tf.reduce_sum(tf.multiply(self.w,2) * reshaped_inputs, axis = -1, keepdims = False)

        #Now I need to add the bias. The bias has shape (num_layers,num_qubits) so I need to tile it to have shape 
        # (batch_size, num_layers, num_qubits)
        tiled_up_bias = tf.reshape(tf.tile(self.b, multiples = [batch_dim, 1]), (batch_dim, self.num_layers, self.num_qubits))
        inputs_times_weights_plus_b = inputs_times_weights + tf.multiply(tiled_up_bias,2)
        reshaped_inputs_times_weights_plus_b = tf.reshape(inputs_times_weights_plus_b, (batch_dim, self.num_layers * self.num_qubits))

        # The activation function is applied to the inputs_times_w_plus_b tensor
        activated_inputs = tf.keras.layers.Activation(self.activation)(reshaped_inputs_times_weights_plus_b)

        # The thetas and the activated inputs are concatenated and then the indices are gathered
        joined_vars = tf.concat([tiled_up_thetas, activated_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        # The computation layer is applied to the circuits and the variables
        output = self.computation_layer([tiled_up_circuits, joined_vars])
        
        return output


        


