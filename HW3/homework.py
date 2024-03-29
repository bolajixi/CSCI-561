import csv
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output

    def backward(self, X, y, learning_rate):
        # Back-Propagation
        error = y - self.output

        # Calculate gradients
        delta_output = error * self.sigmoid_derivative(self.output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate, batch_size):
        data_size = len(X)
        for epoch in range(epochs):
            for i in range(0, data_size, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            if epoch % 100 == 0:        # Print error every 100 epochs
                print(f'Epoch {epoch}: Error {np.mean(np.square(y - self.predict(X)))}')

    def predict(self, X):
        return self.forward(X)

# Core Algorithm
if __name__ == "__main__":
    
    for data_set in range(1, 2):
        print(f"## -- Data Set {data_set}")
        # Load pre-split training data

        ## Train Data ~ 70%
        with open(f"train_data{data_set}.csv", newline='') as train_data_file:
            reader = csv.reader(train_data_file)
            X_train = np.array(list(reader), dtype=float)

        with open(f"train_label{data_set}.csv", newline='') as train_label_file:
            reader = csv.reader(train_label_file)
            y_train = np.array(list(reader), dtype=float)

        ## Test Data ~ 30%
        with open(f"test_data{data_set}.csv", newline='') as test_data_file:
            reader = csv.reader(test_data_file)
            X_test = np.array(list(reader), dtype=float)

        with open(f"test_label{data_set}.csv", newline='') as test_label_file:
            reader = csv.reader(test_label_file)
            y_test = np.array(list(reader), dtype=float)

        # Data description
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Initialize MLP
        mlp = MLP(input_size=2, hidden_size=4, output_size=1)

        # Train the MLP
        mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1, batch_size=2)

        # Predict
        predictions = mlp.predict(X_test)
        result = zip(predictions, y_test)

        print("Predictions vs Actual:")
        for prediction, actual in result:
            print(f"Prediction: {prediction:.2f}, --- Actual: {actual:.2f}")