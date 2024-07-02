import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Dense

# Load the dataset
abalone = fetch_ucirepo(id=1)

# Compute the Age
abalone.data.targets = abalone.data.targets + 1.5

# Determine categorical and numerical features
numerical_ix = abalone.data.features.select_dtypes(include=['float64', 'int32']).columns
categorical_ix = abalone.data.features.select_dtypes(include=['object']).columns

# Transforming the Columns
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), categorical_ix),
                        ("scaler", MinMaxScaler(), numerical_ix)])
new_data = ct.fit_transform(abalone.data.features)
X, y = new_data, abalone.data.targets

# Data Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to define and train perceptron model
def train_perceptron(X_train, y_train, X_val, y_val, epochs):
    model = Sequential()
    model.add(Input((X_train.shape[1],)))
    model.add(Dense(1)) # Perceptron has only one output node without activation
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
    
    train_errors = history.history['mse']
    val_errors = history.history['val_mse']
    
    return train_errors, val_errors

# Function to define and train neural network models
def train_model(X_train, y_train, X_val, y_val, epochs, num_hidden_layers, num_nodes, activation):

    model = Sequential()
    model.add(Input((X_train.shape[1],)))
    model.add(Dense(num_nodes, activation=activation))
    for _ in range(num_hidden_layers):
        model.add(Dense(num_nodes, activation=activation))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
    
    train_errors = history.history['mse']
    val_errors = history.history['val_mse']
    
    return train_errors, val_errors

# Function to plot training and validation errors
def plot_errors(train_errors, val_errors, title):
    plt.plot(train_errors, label='Training MSE')
    plt.plot(val_errors, label='Validation MSE')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

# Model 1: Perceptron Model
train_errors_1, val_errors_1 = train_perceptron(X_train, y_train, X_val, y_val, epochs=50)
plot_errors(train_errors_1, val_errors_1, 'Model 1: Perceptron Model - No hidden layer')

# Model 2: ReLU activation
train_errors_2, val_errors_2 = train_model(X_train, y_train, X_val, y_val, epochs=60, num_hidden_layers=2, num_nodes=20, activation='relu')
plot_errors(train_errors_2, val_errors_2, 'Model 2: ReLU activation, two hidden layers')

# Model 3:Sigmoid activation
train_errors_3, val_errors_3 = train_model(X_train, y_train, X_val, y_val, epochs=80, num_hidden_layers=1, num_nodes=32, activation='sigmoid')
plot_errors(train_errors_3, val_errors_3, 'Model 3: Sigmoid activation, one hidden layer')

# Model 4: Sigmoid activation, two hidden layers
train_errors_4, val_errors_4 = train_model(X_train, y_train, X_val, y_val, epochs=60, num_hidden_layers=2, num_nodes=20, activation='sigmoid')
plot_errors(train_errors_4, val_errors_4, 'Model 4: Sigmoid activation, two hidden layers')


# Record minimum validation errors
min_val_errors = [np.min(val_errors_1), np.min(val_errors_2), np.min(val_errors_3), np.min(val_errors_4)]
min_train_errors = [np.min(train_errors_1), np.min(train_errors_2), np.min(train_errors_3), np.min(train_errors_4)]

# Display minimum Training errors in a table
print("Model\t\tMinimum Training Error")
print("1:Perceptron Model\t\t{:.3f}".format(min_train_errors[0]))
print("2:ReLU activation\t\t{:.3f}".format(min_train_errors[1]))
print("3:Sigmoid activation\t\t{:.3f}".format(min_train_errors[2]))
print("4:Sigmoid activation\t\t{:.3f}".format(min_train_errors[3]))
print("=========================================")

# Display minimum validation errors in a table
print("Model\t\tMinimum Validation Error")
print("1:Perceptron Model\t\t{:.3f}".format(min_val_errors[0]))
print("2:ReLU activation\t\t{:.3f}".format(min_val_errors[1]))
print("3:Sigmoid activation\t\t{:.3f}".format(min_val_errors[2]))
print("4:Sigmoid activation\t\t{:.3f}".format(min_val_errors[3]))
print("=========================================")

# Classification Task

df = fetch_openml(data_id=40983, as_frame=True)

X_2 = df.data
y_2 = df.target

# Convert labels to binary format
y_binary = (y_2 == '1').astype(int)

# Data Split
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_2, y_binary, test_size=0.2, random_state=42)

# Function to build and train a neural network model
def train_model(X_train2, y_train2, X_val2, y_val2, epochs, num_hidden_layers, num_nodes, activation):
    model2 = Sequential()
    model2.add(Input((X_train2.shape[1],)))
    model2.add(Dense(num_nodes, activation=activation))

    for _ in range(num_hidden_layers):
        model2.add(Dense(num_nodes + 5, activation=activation))
    model2.add(Dense(len(set(y_train2)), activation='softmax')) # Output layer
    
    model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model2.fit(X_train2, y_train2, epochs=epochs, validation_data=(X_val2, y_val2), verbose=0)
    
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    return train_accuracy, val_accuracy
    
# Function to plot training and validation errors
def plot_errors(train_accuracy, val_accuracy, title):
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Model 1: ReLU activation
train_accuracy_1, val_accuracy_1 = train_model(X_train2, y_train2, X_val2, y_val2, epochs=60, num_hidden_layers=1, num_nodes=30, activation='relu')
plot_errors(train_accuracy_1, val_accuracy_1, 'Model 1: ReLU activation, one hidden layers')

# Model 2: ReLU activation
train_accuracy_2, val_accuracy_2 = train_model(X_train2, y_train2, X_val2, y_val2, epochs=80, num_hidden_layers=2, num_nodes=42, activation='relu')
plot_errors(train_accuracy_2, val_accuracy_2, 'Model 2: ReLU activation, two hidden layers')

# Model 3:Sigmoid activation
train_accuracy_3, val_accuracy_3 = train_model(X_train2, y_train2, X_val2, y_val2, epochs=60, num_hidden_layers=1, num_nodes=30, activation='sigmoid')
plot_errors(train_accuracy_3, val_accuracy_3, 'Model 3: Sigmoid activation, one hidden layer')

# Model 4: Sigmoid activation, two hidden layers
train_accuracy_4, val_accuracy_4 = train_model(X_train2, y_train2, X_val2, y_val2, epochs=80, num_hidden_layers=2, num_nodes=42, activation='sigmoid')
plot_errors(train_accuracy_4, val_accuracy_4, 'Model 4: Sigmoid activation, two hidden layers')


# Record minimum validation errors
max_val_accuracy = [np.max(val_accuracy_1), np.max(val_accuracy_2), np.max(val_accuracy_3), np.max(val_accuracy_4)]
max_train_accuracy = [np.max(train_accuracy_1), np.max(train_accuracy_2), np.max(train_accuracy_3), np.max(train_accuracy_4)]

# Display maximum Training Accuracy in a table
print("Model\t\tMaximum Training Accuracy")
print("1:ReLU activation Model\t\t\t{:.3f}".format(max_train_accuracy[0]))
print("2:ReLU activation 2HL\t\t\t{:.3f}".format(max_train_accuracy[1]))
print("3:Sigmoid activation\t\t\t{:.3f}".format(max_train_accuracy[2]))
print("4:Sigmoid activation 2HL\t\t{:.3f}".format(max_train_accuracy[3]))
print("=========================================")

# Display maximum validation Accuracy in a table
print("Model\t\tMaximum Validation Accuracy")
print("1:ReLU activation Model\t\t\t{:.3f}".format(max_val_accuracy[0]))
print("2:ReLU activation 2HL\t\t\t{:.3f}".format(max_val_accuracy[1]))
print("3:Sigmoid activation\t\t\t{:.3f}".format(max_val_accuracy[2]))
print("4:Sigmoid activation 2HL\t\t{:.3f}".format(max_val_accuracy[3]))
print("=========================================")