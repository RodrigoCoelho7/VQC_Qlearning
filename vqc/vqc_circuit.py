import cirq
import sympy
import numpy as np
from vqc.vqc_operations import OPERATIONS

"""
This script will contain the class that implements a VQC circuit according to:
1. The number of qubits
2. The number of layers
3. The circuit architecture
4. The data-encoding technique
5. If there is Data Reuploading or Not (Either Baseline or "basic" or "schuld")
6. The measurement technique (Either "lastZ" or "ZZ")

"""

class VQCCIRCUIT():
    def __init__(self, num_qubits, num_layers, circuit_arch, data_reuploading = "baseline", measurement = "ZZ"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.circuit_arch = circuit_arch
        self.data_reuploading = data_reuploading
        self.measurement = measurement
        self.circuit = cirq.Circuit()

        # Create the qubits for the quantum circuit,
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)

        self.inputs = self.generate_input_symbols()
        self.params_per_qubit = self._params_per_qubit()
        self.parameters = self.generate_parameter_symbols()

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()


    def generate_input_symbols(self):
        """
        This function will generate the input symbols for the quantum circuit based on the data reuploading technique
        """
        if self.circuit_arch == "uqc":
            inputs = sympy.symbols(f'y_(0:{self.num_layers*self.num_qubits})')
            inputs = np.asarray(inputs).reshape((self.num_layers, self.num_qubits))
            return inputs
        elif self.data_reuploading == "baseline":
            inputs = sympy.symbols(f'x_(0:{self.num_qubits})')
            inputs = np.asarray(inputs).reshape((1, self.num_qubits))
        elif self.data_reuploading == "basic":
            inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
            inputs = np.asarray(inputs).reshape((self.num_layers, self.num_qubits))
        elif self.data_reuploading == "schuld":
            inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
            inputs = np.asarray(inputs).reshape((self.num_layers, self.num_qubits))
        else:
            raise ValueError("Invalid data reuploading technique!")
        return inputs
    
    def _params_per_qubit(self):
        if self.circuit_arch == "skolik":
            return 2
        elif self.circuit_arch == "lock":
            return 3
        elif self.circuit_arch == "uqc":
            return 1
        else:
            raise ValueError("Invalid circuit architecture!")
    
    def generate_parameter_symbols(self):
        if self.circuit_arch == "uqc":
            params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
            params = np.asarray(params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))
            return params
        if self.data_reuploading == "baseline":
            params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
            params = np.asarray(params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))
        elif self.data_reuploading == "basic":
            params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
            params = np.asarray(params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))
        elif self.data_reuploading == "schuld":
            params = sympy.symbols(f'theta(0:{self.params_per_qubit*(self.num_layers + 1)*self.num_qubits})')
            params = np.asarray(params).reshape((self.num_layers+1, self.num_qubits, self.params_per_qubit))
        return params
    
    def build_skolik_baseline(self):
        #Encoding Layer
        self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
        if self.measurement == "ZZ":
            for l in range(self.num_layers):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
        else:
            for l in range(self.num_layers - 1):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
            #Last Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[self.num_layers-1,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.cnot_chain(self.qubits)
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
  
    def build_skolik_basic(self):
        for l in range(self.num_layers - 1):
            #Encoding Layer
            self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
            #Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.entangling_layer_skolik(self.qubits)
        self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[self.num_layers-1,i], q) for i,q in enumerate(self.qubits))
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[self.num_layers-1,i]) for i,q in enumerate(self.qubits))
        if self.measurement == "lastZ":
            self.circuit += self.operations.cnot_chain(self.qubits)
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
  
    def build_skolik_schuld(self):
        if self.measurement == "ZZ":
            for l in range(self.num_layers):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
                #Encoding Layer
                self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
            #Last Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[self.num_layers,i]) for i,q in enumerate(self.qubits))
        else:
            for l in range(self.num_layers):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
                #Encoding Layer
                self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
            #Last Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.parameters[self.num_layers,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.cnot_chain(self.qubits)
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
  
    def build_lock_baseline(self):
        #Encoding Layer
        self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
        for l in range(self.num_layers):
            #Variational layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.entangling_layer_lock(self.qubits)
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
  
    def build_lock_basic(self):
        for l in range(self.num_layers):
            #Encoding Layer
            self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
            #Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.entangling_layer_lock(self.qubits)
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
  
    def build_lock_schuld(self):
        for l in range(self.num_layers):
            #Variational layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.parameters[l,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.entangling_layer_lock(self.qubits)
            #Encoding Layer
            self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
        #Last Variational Layer
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.parameters[self.num_layers,i]) for i,q in enumerate(self.qubits))
        return self.circuit, list(self.parameters.flat), list(self.inputs.flat)
    
    def build_uqc(self):
        self.parameters = list(self.parameters.flat)
        self.inputs = list(self.inputs.flat)
        for l in range(self.num_layers):
            #Variational layer
            self.circuit += cirq.Circuit(self.operations.uqc(q , self.parameters[l], self.inputs[l]) for i,q in enumerate(self.qubits))
        return self.circuit, self.parameters, self.inputs
    
    def generate_circuit(self):
        if self.circuit_arch == "uqc":
            return self.build_uqc()
        elif self.circuit_arch == "skolik":
          if self.data_reuploading == "baseline":
            return self.build_skolik_baseline()
          elif self.data_reuploading == "basic":
            return self.build_skolik_basic()
          elif self.data_reuploading == "schuld":
            return self.build_skolik_schuld()
        elif self.circuit_arch == "lock":
          if self.data_reuploading == "baseline":
            return self.build_lock_baseline()
          elif self.data_reuploading == "basic":
            return self.build_lock_basic()
          elif self.data_reuploading == "schuld":
            return self.build_lock_schuld()