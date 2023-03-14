
import pennylane as qml
from fourier_analysis.pennylane_operations import QMLOperations
from pennylane.fourier import coefficients, circuit_spectrum

class Fourier_Analysis():
    def __init__(self, num_qubits, num_layers, circuit_arch, data_reuploading, measurement, weights, data):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.circuit_arch = circuit_arch
        self.data_reuploading = data_reuploading
        self.measurement = measurement
        self.weights = weights
        self.data = data
        self.circuit = self.create_circuit()

        #The weights must have shape(num_layers, num_qubits*2)

    def create_circuit(self):
        dev = qml.device("default.qubit", wires = self.num_qubits)
        if self.circuit_arch == "skolik":
            if self.data_reuploading == "schuld":
                circuit = qml.QNode(QMLOperations().schuld_datareup(self.weights, self.data, self.num_layers), dev)
        return circuit
        
