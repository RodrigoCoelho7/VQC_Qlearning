import cirq
import numpy as np
import sympy
from vqc.vqc_operations import OPERATIONS

class VQC():
    def __init__(self, num_qubits,num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.circuit = cirq.Circuit()
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
    

class SkolikBaseline(VQC):
    def __init__(self, num_qubits, num_layers ):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((1, self.num_qubits))

        self.params_per_qubit = 2

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
        for l in range(self.num_layers):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.params[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class SkolikBasic(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((self.num_layers, self.num_qubits))

        self.params_per_qubit = 2

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        for l in range(self.num_layers - 1):
            #Encoding Layer
            self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
            #Variational Layer
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.params[l,i]) for i,q in enumerate(self.qubits))
            self.circuit += self.operations.entangling_layer_skolik(self.qubits)
        self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[self.num_layers-1,i], q) for i,q in enumerate(self.qubits))
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.params[self.num_layers-1,i]) for i,q in enumerate(self.qubits))
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class SkolikSchuld(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((self.num_layers, self.num_qubits))

        self.params_per_qubit = 2

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*(self.num_layers + 1)*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers+1, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        for l in range(self.num_layers):
                #Variational layer
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.params[l,i]) for i,q in enumerate(self.qubits))
                self.circuit += self.operations.entangling_layer_skolik(self.qubits)
                #Encoding Layer
                self.circuit += cirq.Circuit(self.operations.skolik_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
        #Last Variational Layer
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_skolik(q , self.params[self.num_layers,i]) for i,q in enumerate(self.qubits))
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class LockwoodBaseline(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((1, self.num_qubits))

        self.params_per_qubit = 3

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[0,i], q) for i,q in enumerate(self.qubits))
        for l in range(self.num_layers):
                #Variational layer
                self.circuit += self.operations.entangling_layer_lock(self.qubits)
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.params[l,i]) for i,q in enumerate(self.qubits))
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class LockwoodBasic(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((self.num_layers, self.num_qubits))

        self.params_per_qubit = 3

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        for l in range(self.num_layers - 1):
            #Encoding Layer
            self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
            #Variational Layer
            self.circuit += self.operations.entangling_layer_lock(self.qubits)
            self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.params[l,i]) for i,q in enumerate(self.qubits))
        self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[self.num_layers-1,i], q) for i,q in enumerate(self.qubits))
        self.circuit += self.operations.entangling_layer_lock(self.qubits)
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.params[self.num_layers-1,i]) for i,q in enumerate(self.qubits))
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class LockwoodSchuld(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'x(0:{self.num_layers})' + f'_(0:{self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((self.num_layers, self.num_qubits))

        self.params_per_qubit = 3

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*(self.num_layers + 1)*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers+1, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        for l in range(self.num_layers):
                #Variational layer
                self.circuit += self.operations.entangling_layer_lock(self.qubits)
                self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.params[l,i]) for i,q in enumerate(self.qubits))
                #Encoding Layer
                self.circuit += cirq.Circuit(self.operations.lock_encoding(self.inputs[l,i], q) for i,q in enumerate(self.qubits))
        #Last Variational Layer
        self.circuit += self.operations.entangling_layer_lock(self.qubits)
        self.circuit += cirq.Circuit(self.operations.parametrized_gates_lock(q , self.params[self.num_layers,i]) for i,q in enumerate(self.qubits))
        return  self.circuit, list(self.params.flat), list(self.inputs.flat)
    
class UQC(VQC):
    def __init__(self, num_qubits, num_layers):
        super().__init__(num_qubits, num_layers)
        
        self.inputs = sympy.symbols(f'y_(0:{self.num_layers*self.num_qubits})')
        self.inputs = np.asarray(self.inputs).reshape((self.num_layers, self.num_qubits))

        self.params_per_qubit = 1

        self.params = sympy.symbols(f'theta(0:{self.params_per_qubit*self.num_layers*self.num_qubits})')
        self.params = np.asarray(self.params).reshape((self.num_layers, self.num_qubits, self.params_per_qubit))

        self.operations = OPERATIONS()

        self.circuit, self.parameters, self.inputs = self.generate_circuit()

    def generate_circuit(self):
        self.params = list(self.params.flat)
        self.inputs = list(self.inputs.flat)
        for l in range(self.num_layers):
            #Variational layer
            self.circuit += cirq.Circuit(self.operations.uqc(q , self.params[l], self.inputs[l]) for i,q in enumerate(self.qubits))
        return self.circuit, self.params, self.inputs

