import csv
import numpy as np
import pandas as pd

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.w1 = np.random.randn(self.hidden_size, self.input_size)                        # weights_input_hidden
        self.b1 = np.random.randn(self.hidden_size, 1)                                      # bias_input_hidden
        self.w2 = np.random.randn(self.output_size, self.hidden_size)                       # weights_hidden_output
        self.b2 = np.random.randn(self.output_size, 1)                                      # bias_hidden_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

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
        self.z1 = np.dot(self.w1, X) + self.b1
        self.hidden_output = self.relu(self.z1)

        self.z2 = np.dot(self.w2, self.hidden_output) + self.b2
        self.output = self.softmax(self.z2)
        return self.output

    def backward(self, X, y, learning_rate):
        m = y.size

        # Calculate gradients
        dZ2 = self.output - y                                                                # error
        dW2 = 1 / m * np.dot(dZ2, self.hidden_output.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = np.dot(self.w2.T, dZ2) * self.relu_derivative(self.z1)
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1)

        # Update weights and biases
        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

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


    # Remove outliers (i.e houses that cost more than $100m)
    X_train_copy = X_train_copy[X_train_copy['PRICE'] <= 10**7]
    y_filtered = y_train.loc[X_train_copy.index]
    processed_Y_train = y_filtered.to_numpy()

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

        mlp = MLP(input_size= processed_X_train.shape[0], hidden_size=10, output_size=1)

        mlp.train(processed_X_train, processed_Y_train, epochs=1000, learning_rate=0.01, batch_size=100)
        predictions = mlp.predict(processed_X_test)

        result = zip(predictions.flatten(), y_test.values.flatten())

        print("Predictions vs Actual:")
        for count, (prediction, actual) in enumerate(result):
            print(f"Prediction: {prediction:.2f}, --- Actual: {actual:.2f}")

            if count == 10:     # Print first predictions
                break

        print("\n---------------------------------------\n")