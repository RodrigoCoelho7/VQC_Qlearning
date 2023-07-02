
import pennylane as qml
from pennylane import numpy as np

class QMLOperations():
    def __init__(self):
        pass

    def skolik_variational_layer(self,wires, params):
        [[qml.RY(params[2*i], wires[i]),
         qml.RZ(params[(i*2)+1], wires[i])] for i in range(len(wires))]
    
    def skolik_entangling_layer(self,wires):
        [qml.CZ(wires = [i,j]) for i,j in zip(wires, wires[1:])]
        if len(wires) != 2:
            qml.CZ(wires = [wires[0], wires[-1]])
    
    def skolik_data_encoding(self,wires, data, data_dim):
        if data_dim == 1:
            [qml.RX(data, wires[i], id = f"x_{i}") for i in range(len(wires))]
        else:
            [qml.RX(data[i], wires[i], id = f"x_{i}") for i in range(len(wires))]

    def uqc_layer(self, wires, data, rotational_weights, input_weights, bias_weights):
        [qml.RZ(np.dot(2 * input_weights[i], data) + bias_weights[i] , wires[i]) for i in range(len(wires))]
        [qml.RY(2 * rotational_weights[i], wires[i]) for i in range(len(wires))]

    def schuld_datareup(self, params,num_layers, data_dim, data):
        for l in range(num_layers):
            self.skolik_variational_layer(range(4), params[l])
            self.skolik_entangling_layer(range(4))
            self.skolik_data_encoding(range(4), data, data_dim)
        self.skolik_variational_layer(range(4), params[num_layers])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    def basic_datareup(self, params,num_layers,data_dim, data):
        for l in range(num_layers-1):
            self.skolik_data_encoding(range(4), data, data_dim)
            self.skolik_variational_layer(range(4), params[l])
            self.skolik_entangling_layer(range(4))
        self.skolik_data_encoding(range(4), data, data_dim)
        self.skolik_variational_layer(range(4), params[num_layers-1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    def baseline_datareup(self, params,num_layers,data_dim, data):
        self.skolik_data_encoding(range(4), data, data_dim)
        for l in range(num_layers -1):
            self.skolik_variational_layer(range(4), params[l])
            self.skolik_entangling_layer(range(4))
        self.skolik_variational_layer(range(4), params[num_layers-1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def uqc(self, params, num_qubits, num_layers, data):
        rotational_weights = params[0]
        input_weights = params[1]
        bias_weights = params[2]
        for l in range(num_layers):
            self.uqc_layer(range(num_qubits), data, rotational_weights[l], input_weights[l], bias_weights[l])
        if num_qubits == 2:
            return qml.expval(qml.PauliZ(0))
        elif num_qubits == 4:
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))