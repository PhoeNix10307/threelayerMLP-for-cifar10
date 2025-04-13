import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', alpha=0.01):
        """
        Initialize a 3-layer neural network.
        - input_size: number of input features
        - hidden_size: number of neurons in the hidden layer
        - output_size: number of output classes
        - activation: activation function to use
        - alpha: used for leaky_relu and elu
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)

        self.activation = activation.lower()
        self.alpha = alpha

    def _activate(self, z):
        """Apply the activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'leaky_relu':
            return np.where(z > 0, z, self.alpha * z)
        elif self.activation == 'elu':
            return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'swish':
            return z / (1 + np.exp(-z))  # swish = x * sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_derivative(self, a, z):
        """Compute the derivative of the activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(z > 0, 1, self.alpha)
        elif self.activation == 'elu':
            return np.where(z > 0, 1, self.alpha * np.exp(z))
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - a ** 2
        elif self.activation == 'swish':
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid + z * sigmoid * (1 - sigmoid)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, X):
        """Forward pass."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activate(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))  # for numerical stability
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, reg_lambda):
        """Backward pass with cross-entropy + L2 regularization."""
        num_examples = X.shape[0]
        probs = self.probs.copy()
        probs[np.arange(num_examples), y] -= 1
        probs /= num_examples

        dW2 = self.a1.T @ probs + reg_lambda * self.W2
        db2 = np.sum(probs, axis=0)

        delta2 = (probs @ self.W2.T) * self._activate_derivative(self.a1, self.z1)
        dW1 = X.T @ delta2 + reg_lambda * self.W1
        db1 = np.sum(delta2, axis=0)

        return dW1, db1, dW2, db2

    def loss(self, X, y, reg_lambda):
        """Compute softmax cross-entropy loss with L2 regularization."""
        probs = self.forward(X)
        log_probs = -np.log(np.clip(probs[np.arange(X.shape[0]), y], 1e-10, 1.0))
        data_loss = np.mean(log_probs)
        reg_loss = 0.5 * reg_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss
