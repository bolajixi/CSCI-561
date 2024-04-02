import csv
import numpy as np
import pandas as pd
from pandas import DataFrame

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

    def relu(self, x):
        return np.maximum(0, x)

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
                mse = np.mean(np.square(y - self.predict(X)))
                error_percentage = mse * 100
                print(f'Epoch {epoch}: Error {error_percentage:.2f}%')

    def predict(self, X):
        return self.forward(X)

# Helper functions
class Scaler:
    def fit_transform(self, numerical_columns):
        self.mean = np.mean(numerical_columns, axis=0)
        self.std = np.std(numerical_columns, axis=0)
        scaled_numerical_columns = (numerical_columns - self.mean) / self.std
        return scaled_numerical_columns

    def transform(self, numerical_columns):
        scaled_numerical_columns = (numerical_columns - self.mean) / self.std
        return scaled_numerical_columns

def one_hot(y):
    df = DataFrame(y.astype(str), columns=['beds'])
    one_hot_encoded = pd.get_dummies(df, dtype=int)
    return one_hot_encoded, list(one_hot_encoded.columns)

def reverse_one_hot(one_hot_encoded, categories):
    max_indices = np.argmax(one_hot_encoded, axis=1)
    reverse_encoded = [categories[idx] for idx in max_indices]
    return reverse_encoded


# Core Algorithm
if __name__ == "__main__":
    
    for data_set in range(1, 6):
        print(f"\n## -- Data Set {data_set}")
        print("---------------------------------------\n")

        # Load pre-split training data
        X_train = pd.read_csv(f"./testcases/train_data{data_set}.csv")
        y_train = pd.read_csv(f"./testcases/train_label{data_set}.csv")

        X_test = pd.read_csv(f"./testcases/test_data{data_set}.csv")
        y_test = pd.read_csv(f"./testcases/test_label{data_set}.csv")

        ## Data processing
        drop_columns = ['ADDRESS','STATE','MAIN_ADDRESS','STREET_NAME','LONG_NAME','FORMATTED_ADDRESS','LATITUDE','LONGITUDE','LOCALITY','BROKERTITLE']
        X_train_copy = X_train.copy().drop(columns=drop_columns)

        # - Remove outliers (i.e houses that cost more than $100m)
        X_train_copy = X_train_copy.drop(X_train_copy[X_train_copy['PRICE'] > 10**7].index)
        y_train_values = y_train.values.flatten()
        y_train_filtered = y_train.loc[X_train_copy.index]

        # - Encode categorical variables (i.e. BEDS)
        # y_train_encoded, categories = pd.factorize(y_train_filtered.values.flatten())
        one_hot_y, categories = one_hot(y_train_filtered.values.flatten())
        one_hot_y = one_hot_y.to_numpy()

        col_to_scale = ['PRICE','BATH','PROPERTYSQFT']
        col_to_encode = ['TYPE','ADMINISTRATIVE_AREA_LEVEL_2','SUBLOCALITY']

        # - Scale numerical variables (i.e. PRICE, BATH, PROPERTYSQFT)
        scaler = Scaler()
        scaled_numerical_columns = scaler.fit_transform(X_train_copy[col_to_scale])
        X_train_copy[col_to_scale] = scaled_numerical_columns

        X_train_encoded = pd.get_dummies(X_train_copy, columns=col_to_encode, drop_first=False, dtype=int).to_numpy()

        # Data description
        print(f"X_train #{data_set} shape: {X_train_encoded.shape}")
        print(f"y_train #{data_set} shape: {one_hot_y.shape}")

        mlp = MLP(input_size=57, hidden_size=4, output_size=22)
        mlp.train(X_train_encoded, one_hot_y, epochs=1000, learning_rate=0.1, batch_size=100)

        predictions = mlp.predict(X_test)
        predicted_beds = reverse_one_hot(predictions, categories)
        predicted_beds = [prediction.split('beds_')[1] for prediction in predicted_beds]

        result = zip(predictions, y_test.values.flatten())

        print("Predictions vs Actual:")
        for prediction, actual in result:
            print(f"Prediction: {prediction:.2f}, --- Actual: {actual:.2f}")

        print("\n---------------------------------------\n")