# Most of this classifier is inspired and a modified version of the lasagne tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html

import os
import gzip
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from pandas.io.parsers import read_csv

import theano
import theano.tensor as T

import lasagne


# ############################# Load dataset ###############################
def load_dataset():
    # Read training and test sets
    # Preprocessing by creating a cross validation set and reshaping the pixels

    def read_data(filename):
        with gzip.open(filename, 'rb') as f:
                return read_csv(f, header=None)

    train_data = read_data("training_data.csv.gz")
    test_data = read_data("test_data.csv.gz")

    # Test set is already standarized in range [0-1]

    #### Spawn training and cv variables ####
    y_train = np.int32(train_data[train_data.columns[-1]].values)
    X_train = train_data.drop(train_data.columns[-1], 1).values

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    #### Spawn test variables ####
    y_test = np.int32(test_data[test_data.columns[-1]].values)
    X_test = test_data.drop(test_data.columns[-1], 1).values

    #### Unroll pixels ####
    X_train = X_train.reshape(-1,  28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    X_val = X_val.reshape(-1, 28, 28)

    return X_train, y_train, X_val, y_val, X_test, y_test

# ############################# Displaying data ###############################
# Several helper functions for data displaying

def select_random_elements(images, labels, num_elem=10):
    # Randomly selects images and their corresponding labels
    # Returns the selected images and labels
    selected_index = randint(0, len(images), num_elem)
    selected_X, selected_y = zip(*[(images[selected_index[i]], labels[selected_index[i]]) for i in range(len(selected_index))])
    return list(selected_X), selected_y

def display_images(images, labels, num_figure=1):
    # Display images with their corresponding label
    # This function is used both for figure 1 and 3

    assert len(images) == len(labels)
    plt.figure(num_figure)


    # Loop will only work for even numbers
    for i in range(len(labels)):
        plt.subplot(5,2,i)

        # Display image
        plt.imshow(images[i])
        plt.axis('off')

        # Display label
        plt.text(-20, 20, str(labels[i]), fontsize=50) #fontsize and displacement should be scaled with display_amount
        plt.axis('off')
    plt.show()

def display_weights(weights, num_elem=10):
    plt.figure(4)
    selected_index = randint(0, len(weights[0]), num_elem)
    for i in range(len(selected_index)):
        plt.subplot(2,5,i)
        img = weights[:,selected_index[i]].reshape(28,28)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


# ############################# Model building ###############################
# Building the neural network model with lasagne
# This function is a downstripped version of the lasagne tutorial

# Creates an MLP of one hidden layer, followed by
# a softmax output layer of 10 units.
# Provided batchsize will allow theano to do additional optimization

def build_mlp(num_hidden_units=625,
                batchsize=None,
                input_var=None):

    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(batchsize, 28, 28),
                                     input_var=input_var)

    # Hidden layer
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform())

    # Output layer
    l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out, l_hid

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size

def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ############################# Main function ###############################
# The main function calls the above helper functions and training of the neural network

def main(learning_rate=0.05,
            num_hidden_units=625,
            batchsize=500,
            num_epochs=100):

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    sel_X, sel_y = select_random_elements(X_train, y_train)
    display_images(sel_X, sel_y)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    network, hidden_layer = build_mlp(num_hidden_units=num_hidden_units, batchsize=batchsize, input_var=input_var)

    # Create a loss function for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(
            loss, params, learning_rate=learning_rate)

    # Prediction accuracy
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    predicted_val = T.argmax(prediction, axis=1)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [loss, acc])

    # Get prediction
    prediction_fn = theano.function([input_var, target_var], [loss, predicted_val])

    train_errors = np.zeros(num_epochs)
    val_errors = np.zeros(num_epochs)

    # Launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # Full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            train_err += err
            train_batches += 1

        # Full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Store errors for visualization
        train_errors[epoch] = train_err
        val_errors[epoch] = val_err

        # Print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # Compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchsize):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Displaying Training error and validation error:
    # Training Error
    plt.figure(2)
    plt.subplot(211)
    plt.plot(train_errors)
    plt.ylabel("Train error")
    # Validation Error
    plt.subplot(212)
    plt.plot(val_errors)
    plt.ylabel("Validation error")
    plt.show()

    # Displaying prediction:
    selected_X, selected_y = select_random_elements(X_test, y_test, 10)
    err, predictions = prediction_fn(selected_X, selected_y)
    display_images(selected_X, predictions, num_figure=3)

    # Displaying weights:
    params = lasagne.layers.get_all_param_values(hidden_layer)
    weights = np.array(params[0])
    display_weights(weights)

main()
