from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# Create dataset
X, y = make_blobs(n_samples=40, centers=2, n_features=2, random_state=0)
y = 2*y - 1  # convert labels to {-1,1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Parameters
x1 = Parameter("x1")
x2 = Parameter("x2")
theta = Parameter("θ")

# Quantum circuit (2 qubits)
qc = QuantumCircuit(2)
qc.ry(x2, 0)
qc.ry(x1, 1)
qc.cx(0,1)
qc.ry(theta, 0)

# Quantum Neural Network
qnn = EstimatorQNN(
    circuit=qc,
    input_params=[x1, x2],
    weight_params=[theta]
)

# ML model
classifier = NeuralNetworkClassifier(qnn)

# Train
classifier.fit(X_train, y_train)

# Evaluate
score = classifier.score(X_test, y_test)

print("Quantum ML Accuracy:", score)