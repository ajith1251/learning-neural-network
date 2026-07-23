import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Data for AND, OR, XOR gates
gates_data = {
    'AND': (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([0, 0, 0, 1])
    ),
    'OR': (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([0, 1, 1, 1])
    ),
    'XOR': (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([0, 1, 1, 0])
    ),
}

# Train and evaluate perceptron for each gate
for gate, (X, y) in gates_data.items():
    perceptron = Perceptron(max_iter=10, eta0=1, random_state=42)
    perceptron.fit(X, y)
    
    y_pred = perceptron.predict(X)
    acc = accuracy_score(y, y_pred) * 100
    
    print(f"{gate} gate accuracy: {acc:.2f}%")
    print(f"Predictions: {y_pred}")
    print(f"True Labels: {y}\n")
