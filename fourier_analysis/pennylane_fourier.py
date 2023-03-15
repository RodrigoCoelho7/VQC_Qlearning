
import pennylane as qml
from fourier_analysis.pennylane_operations import QMLOperations
from pennylane.fourier import coefficients, circuit_spectrum
from functools import partial
from pennylane.fourier.visualize import *
import matplotlib.pyplot as plt
import numpy as np

class Fourier_Analysis():
    def __init__(self, num_qubits, num_layers, circuit_arch, data_reuploading, measurement, weights, data):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.circuit_arch = circuit_arch
        self.data_reuploading = data_reuploading
        self.measurement = measurement
        self.weights = weights
        self.data = data
        if type(data) == int or type(data) == float:
            self.data_dim = 1
        else:
            self.data_dim = len(data)
        self.operations = QMLOperations()
        self.circuit = self.create_circuit()

        #The weights must have shape(num_layers, num_qubits*2)

    def create_circuit(self):
        dev = qml.device("default.qubit", wires = self.num_qubits)
        if self.circuit_arch == "skolik":
            if self.data_reuploading == "schuld":
                circuit = qml.QNode(self.operations.schuld_datareup, dev)
            elif self.data_reuploading == "basic":
                circuit = qml.QNode(self.operations.basic_datareup, dev)
        return circuit
    
    def draw_circuit(self):
        return print(qml.draw(self.circuit)(self.weights, self.num_layers, self.data_dim, self.data))
    
    def circuit_spectrum(self):
        return circuit_spectrum(self.circuit)(self.weights, self.num_layers,self.data_dim, self.data)
    
    def fourier_coefficients(self, weights = None):
        if weights is None:
            partial_circuit = partial(self.circuit, self.weights, self.num_layers, self.data_dim)
        else:
            partial_circuit = partial(self.circuit, weights, self.num_layers, self.data_dim)
        return coefficients(partial_circuit, self.data_dim, self.num_layers)
    
    def plot_coefficients(self, weights = None):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True) # Set up the axes
        bar(self.fourier_coefficients(weights), self.data_dim, ax)

    def plot_random_coefficients_distribution(self, dist_type = "violin"):
        coeffs = []

        for _ in range(100):
            weights = np.random.uniform(0, 2*np.pi, size=(self.num_layers+1, self.num_qubits*2))
            c = coefficients(partial(self.circuit, weights, self.num_layers, self.data_dim), self.data_dim, self.num_layers)
            coeffs.append(np.round(c, decimals=8))

        if dist_type == "violin":
            fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
            violin(coeffs, self.data_dim, ax, show_freqs=True)

        elif dist_type == "box":
            # The subplot axes must be *polar* for the radial plots
            fig, ax = plt.subplots(
                1, 2, sharex=True, sharey=True,
                subplot_kw=dict(polar=True),
                figsize=(15, 8)
            )
            radial_box(coeffs, 2, ax, show_freqs=True, show_fliers=False)

        elif dist_type == "panel":
            assert(self.data_dim <= 2)
            # Need a grid large enough to hold all coefficients up to frequency 2
            fig, ax = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
            panel(coeffs, 2, ax)



