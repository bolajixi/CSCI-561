import csv
import numpy as np
import pandas as pd
import random

class MLP:
    def __init__(self, layer_size):
        self.input_size = layer_size[0]
        self.output_size = layer_size[-1]
        # self.hidden_size = layer_size[1]

        self.sizes_per_layer = layer_size
        self.num_layers = len(layer_size)

        # Initialize weights and biases for hidden layer --> output layer
        self.weights = [np.random.randn(layer_output_size, layer_input_size)/np.sqrt(layer_input_size)
                        for layer_input_size, layer_output_size in zip(self.sizes_per_layer[:-1], self.sizes_per_layer[1:])]

        self.biases = [np.random.randn(y, 1) for y in self.sizes_per_layer[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0)

    def softmax(self, x):
        # return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass through the network
        for bias, weight in zip(self.biases, self.weights):
            X = self.relu(np.dot(weight, X) + bias)

        self.output = X
        return self.output

    def backward(self, X, y, learning_rate):
        # Forward pass ------------------------------------------------------
        current_activation = X
        all_activations = [X]                       # list to store all the activations, layer by layer
        all_z_vectors = []                          # list to store all the z vectors, layer by layer
        
        for bias, weight in zip(self.biases, self.weights):
            current_z = np.dot(weight, current_activation) + bias
            all_z_vectors.append(current_z)

            current_activation = self.relu(current_z)
            all_activations.append(current_activation)


        # Backward pass ------------------------------------------------------
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        output_error = (all_activations[-1] - y) * self.relu_derivative(all_z_vectors[-1])
        nabla_b[-1] = output_error
        nabla_w[-1] = np.dot(output_error, all_activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = all_z_vectors[-l]
            d_relu = self.relu_derivative(z)
            output_error = np.dot(self.weights[-l+1].transpose(), output_error) * d_relu
            nabla_b[-l] = output_error
            nabla_w[-l] = np.dot(output_error, all_activations[-l-1].transpose())

        return (nabla_b, nabla_w)


    def _update_mini_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        batch_size = len(batch)
        X_batch, y_batch = zip(*batch)

        delta_nabla_b, delta_nabla_w = self.backward(X_batch, y_batch, learning_rate)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases ------------------------------------------
        self.weights = [w-(learning_rate/batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/batch_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, learning_rate, batch_size, test_data=None):
        data_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i:i+batch_size] for i in range(0, data_size, batch_size)]

            for batch in mini_batches:
                self._update_mini_batch(batch, learning_rate)

            if epoch % 100 == 0 and test_data is not None:                                    # Print error every 100 epochs
                X, y = zip(*test_data)

                result = (self.parse_results(self.predict(X)), y)
                num_correct_predictions = sum(int(x == y) for (x, y) in result)

                for count, (prediction, actual) in enumerate(result):
                    print(f"Prediction: {prediction}, --- Actual: {actual}")

                    if count == 10:     # Print first predictions
                        break

    def predict(self, X):
        return np.argmax(self.forward(X))

    def parse_results(self, results):
        # convert one hot encoding to classification category
        return

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

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None
        
    def fit(self, data):
        self.categories_ = {}
        for column in data.columns:
            unique_values = data[column].unique()
            self.categories_[column] = sorted(unique_values)
        
    def transform(self, data):
        if self.categories_ is None:
            raise ValueError("fit method should be called first")
        
        encoded_data = pd.DataFrame()
        for column in data.columns:
            for category in self.categories_[column]:
                encoded_data[column + '_' + str(category)] = (data[column] == category).astype(int)
        
        return encoded_data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# Core functions ------------------------------------------------------------------------------
def load_data(data_set):
    """
    Load pre-split training and testing data for a given data set.
    """
    X_train = pd.read_csv(f"./testcases/train_data{data_set}.csv")
    y_train = pd.read_csv(f"./testcases/train_label{data_set}.csv")
    X_test = pd.read_csv(f"./testcases/test_data{data_set}.csv")
    try:
        y_test = pd.read_csv(f"./testcases/test_label{data_set}.csv")
    except FileNotFoundError:
        y_test = None

    return X_train, y_train, X_test, y_test

def preprocess_data(x_train, y_train, x_test, col_to_scale, col_to_encode, scaler, encoder):
    """
    Preprocess the training data: drop columns, remove outliers, scale numerical variables,
    encode categorical variables, and return the processed data.
    """
    drop_columns = ['ADDRESS', 'STATE', 'MAIN_ADDRESS', 'STREET_NAME', 'LONG_NAME', 
                    'FORMATTED_ADDRESS', 'LATITUDE', 'LONGITUDE', 'LOCALITY', 'BROKERTITLE']
    X_train_copy = x_train.copy().drop(columns=drop_columns)
    X_test_copy = x_test.copy().drop(columns=drop_columns)

    # Scale numerical variables
    X_train_copy[col_to_scale] = scaler.fit_transform(X_train_copy[col_to_scale])
    X_test_copy[col_to_scale] = scaler.transform(X_test_copy[col_to_scale])

    # Encode categorical variables
    X_train_encoded = encoder.fit_transform(X_train_copy[col_to_encode])
    X_test_encoded = encoder.transform(X_test_copy[col_to_encode])

    processed_X_train = pd.concat([X_train_copy, X_train_encoded], axis=1).drop(columns=col_to_encode).to_numpy()
    processed_X_test = pd.concat([X_test_copy, X_test_encoded], axis=1).drop(columns=col_to_encode).to_numpy()
    
    return processed_X_train, processed_Y_train, processed_X_test


# Core Algorithm ------------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_FILE = "output.txt"
    FILE_WRITE_FORMAT = "w"
    
    for data_set in range(1, 6):
        print(f"\n## -- Data Set {data_set}")
        print("---------------------------------------\n")

        X_train, y_train, X_test, y_test = load_data(data_set)

        col_to_scale = ['PRICE','BATH','PROPERTYSQFT']
        col_to_encode = ['TYPE','ADMINISTRATIVE_AREA_LEVEL_2','SUBLOCALITY']

        scaler = Scaler()
        encoder = OneHotEncoder()

        processed_X_train, processed_Y_train, processed_X_test = preprocess_data(X_train, y_train, X_test, col_to_scale, col_to_encode, scaler, encoder)

        processed_X_train = processed_X_train.T
        processed_Y_train = processed_Y_train.T
        processed_X_test = processed_X_test.T

        # Data description
        print(f"X_train #{data_set} shape: {processed_X_train.shape}")
        print(f"y_train #{data_set} shape: {processed_Y_train.shape}")
        print(f"X_test #{data_set} shape: {processed_X_test.shape}")

        network_layer_sizes = [processed_X_train.shape[0], 10, 1]    # input_size, { hidden_size .. }, output_size
        mlp = MLP(layer_size = network_layer_sizes)
        
        training_data = zip(processed_X_train, processed_Y_train)
        test_data = zip(processed_X_test, y_test)

        mlp.train(training_data, epochs=1000, learning_rate=0.01, batch_size=100, test_data=test_data)
        predictions = mlp.predict(processed_X_test)

        result = ['BEDS'] + [str(i) for i in self.parse_results(predictions)]
        result = '\n'.join(result)

        with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
            output_file.write(result + "\n")

        print("\n---------------------------------------\n")