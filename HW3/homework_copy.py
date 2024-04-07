import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layer_size, output_encoder):
        self.input_size = layer_size[0]
        self.output_size = layer_size[-1]
        self.output_encoder = output_encoder

        self.sizes_per_layer = layer_size
        self.num_layers = len(layer_size)

        self.count = 0

        # Initialize weights and biases for hidden layer --> output layer
        self.weights = [np.random.randn(layer_input_size, layer_output_size) / np.sqrt(layer_input_size)
                        for layer_input_size, layer_output_size in zip(self.sizes_per_layer[:-1], self.sizes_per_layer[1:])]

        self.biases = [np.random.randn(1, layer_size) for layer_size in self.sizes_per_layer[1:]]

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
        # Forward pass through the networks (for prediction)
        for bias, weight in zip(self.biases, self.weights):
            X = self.sigmoid(np.dot(X, weight) + bias)

        self.output = X
        return self.output

    def backward(self, X, y):
        # Forward pass ------------------------------------------------------
        current_activation = X
        all_activations = [X]                       # list to store all the activations, layer by layer
        all_z_vectors = []                          # list to store all the z vectors, layer by layer
        
        for bias, weight in zip(self.biases, self.weights):
            current_z = np.dot(current_activation, weight) + bias
            all_z_vectors.append(current_z)

            current_activation = self.sigmoid(current_z)
            all_activations.append(current_activation)


        # Backward pass ------------------------------------------------------
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        output_error = (all_activations[-1] - y) * self.sigmoid_derivative(all_z_vectors[-1])
        nabla_b[-1] = output_error
        nabla_w[-1] = np.dot(all_activations[-2].transpose(), output_error)

        for l in range(2, self.num_layers):
            z = all_z_vectors[-l]
            d_sigmoid = self.sigmoid_derivative(z)
            output_error = np.dot(output_error, self.weights[-l+1].transpose()) * d_sigmoid
            nabla_b[-l] = output_error
            nabla_w[-l] = np.dot(all_activations[-l-1].transpose(), output_error)

        return (nabla_b, nabla_w)


    def _update_mini_batch(self, X_batch, y_batch, learning_rate):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        batch_size = len(X_batch)

        for sample_index in range(X_batch.shape[0]):
            x_sample = X_batch[sample_index].reshape(1, -1)
            y_sample = y_batch[sample_index].reshape(1, -1)

            delta_nabla_b, delta_nabla_w = self.backward(x_sample, y_sample)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases ------------------------------------------
        self.weights = [w-(learning_rate/batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/batch_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, learning_rate, batch_size, test_data=None):
        data_size = len(training_data[0])
        accuracies = []

        for epoch in range(epochs):
            # Shuffle training_data
            X_train, y_train = training_data

            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            X_batches = [ X_train[i:i+batch_size] for i in range(0, data_size, batch_size) ]
            y_batches = [ y_train[i:i+batch_size] for i in range(0, data_size, batch_size) ]

            # Remove last batch if it is smaller than batch_size (e.g. if data_size is not divisible by batch_size)
            if X_batches[-1].shape[0] != batch_size:
                X_batches = X_batches[:-1]
                y_batches = y_batches[:-1]

            for X_batch, y_batch in zip(X_batches, y_batches):
                self._update_mini_batch(X_batch, y_batch, learning_rate)                

            if epoch % 100 == 0 and test_data is not None:                                    # Print error every 100 epochs
                X, y = test_data

                predictions = ( self.output_encoder.inverse_transform('BEDS', self.predict(X)) ).tolist()
                y = y.flatten().tolist()

                result = zip(predictions, y)
                num_correct_predictions = sum(int(x == y) for x, y in result)
                accuracy = num_correct_predictions / len(y)
                accuracies.append(accuracy)

                print(f"Number of correct predictions: {num_correct_predictions}")
                print(f"Epoch {epoch}: Accuracy: {accuracy}")
        
        # Plotting accuracy
        plt.plot(range(0, epochs, 100), accuracies, label='Accuracy Plot')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def predict(self, X):
        predictions_idx = []

        for sample_index in range(X.shape[0]):
            x_sample = X[sample_index].reshape(1, -1)
            predictions_idx.append(np.argmax(self.forward(x_sample)[0]))

        predictions_idx = np.array(predictions_idx)
        return predictions_idx

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

    def inverse_transform(self, category, encoded_data):
        if self.categories_ is None:
            raise ValueError("fit method should be called first")

        num_samples = encoded_data.shape[0]
        original_data = np.zeros((num_samples,), dtype=int)
        
        for i in range(num_samples):
            index = int(encoded_data[i])
            original_data[i] = self.categories_[category][index]
        
        return original_data

class LabelEncoder:
    def __init__(self):
        self.mapping = {}

    def fit(self, series):
        unique_values = series.unique()
        for index, value in enumerate(unique_values):
            self.mapping[value] = index

    def transform(self, series):
        return series.map(self.mapping)

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

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

def preprocess_data(x_train, y_train, x_test, y_test, col_to_scale, col_to_encode, scaler, encoders):
    """
    Preprocess the training data: drop columns, remove outliers, scale numerical variables,
    encode categorical variables, and return the processed data.
    """
    drop_columns = ['ADDRESS', 'STATE', 'MAIN_ADDRESS', 'STREET_NAME', 'LONG_NAME', 
                    'FORMATTED_ADDRESS', 'LATITUDE', 'LONGITUDE', 'LOCALITY', 'BROKERTITLE']
    X_train_copy = x_train.copy().drop(columns=drop_columns)
    X_test_copy = x_test.copy().drop(columns=drop_columns)

    x_encoder, y_encoder = encoders

    # Scale numerical variables
    X_train_copy[col_to_scale] = scaler.fit_transform(X_train_copy[col_to_scale])
    X_test_copy[col_to_scale] = scaler.transform(X_test_copy[col_to_scale])

    # Encode categorical variables -- (X)
    X_train_encoded = x_encoder.fit_transform(X_train_copy[col_to_encode])
    X_test_encoded = x_encoder.transform(X_test_copy[col_to_encode])

    processed_X_train = pd.concat([X_train_copy, X_train_encoded], axis=1).drop(columns=col_to_encode).to_numpy()
    processed_X_test = pd.concat([X_test_copy, X_test_encoded], axis=1).drop(columns=col_to_encode).to_numpy()

    # Encode categorical variables -- (Y)
    processed_Y_train = y_encoder.fit_transform(y_train)
    processed_Y_test = None

    if y_test is not None:
        processed_Y_test = y_encoder.transform(y_test)
        return processed_X_train, processed_Y_train, processed_X_test, processed_Y_test
    
    return processed_X_train, processed_Y_train, processed_X_test, None


# Core Algorithm ------------------------------------------------------------------------------
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"
total_elapsed_minutes = 0

for data_set in range(1, 6):
    start_time = time.time()
    print(f"\n## -- Data Set {data_set}")
    print("---------------------------------------\n")

    X_train, y_train, X_test, y_test = load_data(data_set)

    col_to_scale = ['PRICE','BATH','PROPERTYSQFT']
    col_to_encode = ['TYPE','ADMINISTRATIVE_AREA_LEVEL_2','SUBLOCALITY']

    scaler = Scaler()
    x_encoder = OneHotEncoder()
    y_encoder = OneHotEncoder()

    encoders = [x_encoder, y_encoder]

    processed_X_train, processed_Y_train, processed_X_test, processed_Y_test = preprocess_data(X_train, y_train, X_test, y_test, col_to_scale, col_to_encode, scaler, encoders)

    # Data description
    print(f"X_train #{data_set} shape: {processed_X_train.shape}")
    print(f"y_train #{data_set} shape: {processed_Y_train.shape}")
    print(f"X_test #{data_set} shape: {processed_X_test.shape}")
    print(f"y_test #{data_set} shape: {processed_Y_test.shape}\n")

    network_layer_sizes = [processed_X_train.shape[1], 100, processed_Y_train.shape[1]]    # input_size, { hidden_size .. }, output_size
    mlp = MLP(layer_size= network_layer_sizes, output_encoder=y_encoder)
    
    training_data = (processed_X_train, processed_Y_train.to_numpy())
    test_data = (processed_X_test, y_test.to_numpy())

    mlp.train(training_data, epochs=1000, learning_rate=0.01, batch_size=32, test_data=test_data)
    predictions = mlp.predict(processed_X_test)

    result = ['BEDS'] + [str(i) for i in y_encoder.inverse_transform('BEDS', predictions)]
    result = '\n'.join(result)

    with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
        output_file.write(result + "\n")

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    total_elapsed_minutes += minutes
    total_elapsed_minutes += seconds / 60

    print(f"\n\nElapsed Time = {minutes} minutes and {seconds} seconds")

    print("\n---------------------------------------\n")

print(f"\n\nTotal Elapsed Time Across All Data Sets = {total_elapsed_minutes} minutes")

# Done: Improve hidden layer neuron size from 10 to 80 (network can capture more features)
# Done: Add longitude and latitude (provide more information) -- this does not seem to improve accuracy
# TODO: Update activation from sigmoid to relu / leaky relu to improve convergence