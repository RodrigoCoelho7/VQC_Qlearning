
import pennylane as qml

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
