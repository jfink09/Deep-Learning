
# For the sequential model, import Sequential and Dense

from keras.models import Sequential
from keras.layers import Dense

# Sequential models have a single input and a single output
# The first numbers in the Dense() is the number of units/neurons
# The input shape of 784 specifies 784 elements

model = Sequential([
     Dense(32,activation="relu",input_shape=(784,)),
     Dense(10,activation="softmax"),
])

# Display the model
model.summary()

# Sequential model build with .add() to add layers

model = Sequential()

model.add(Dense(32,activation="relu",input_shape=(784,)))
model.add(Dense(10,activation="softmax"))

# Displays the same model
model.summary()

# Can use either input_shape() or input_dim

# input layer is a tensor, not a layer. It is the starting tensor sent to the first hidden layer.
# The starting tensor needs to have the same shape as the training data.

model = Sequential()

model.add(Dense(units=32, activation="relu", input_shape=(784,))) # Do not need to use units=, can use only the value for number of neurons

model.summary()

# Input_dim has only one dimension (scalar number, number of elements), it does not need to be a tuple.

model = Sequential()

model.add(Dense(32, activation="relu", input_dim=784))

model.summary()

# Before the model is trained, a learning process is configured with the compile method.
# The compile method takes in three arguments (an optimizer, a loss function, and a list of metrics).
# The optimizer can be a string identifier of an existing optimizer like rmsprop or adagrad or an instance of an Optimizer class.
# The loss function is the function that the model tries to minimize.
# The loss function can be a string identifier of an existing loss function like categorial_crossentropy or mse or an objective function.
# For any classification problem, set the list of metrics to metrics=["accuracy"]. It can be the string identifier of an existing metric or a custom metric function.
# Optimizers and losses are always needed for learning, but metrics are not always required.

# For multi-class classification problems
model.compile(optimizer="rmsprop",loss="categorial_crossentropy",metrics=["accuracy"])

# For binary classification problems
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

# For mean squared error regression problems
model.compile(optimizer="rmsprop",loss="mse")

# For custom metrics

import keras.backend as K

def mean_pred(y_true,y_pred):
  return K.mean(y_pred)

model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy",mean_pred])

# Training
# Keras models are trained on NumPy arrays of input data and labels
# Can use three models fit function, fit_generator, train_on_batch
# fit function is basic
# fit_generator for large datasets which takes in a generator instead of a NumPy array,
# train_on_batch for a single gradient update over one batch of samples.

# Make a model
model = Sequential()

# Add layers
model.add(Dense(32,activation="relu",input_dim=100))
model.add(Dense(1,activation="sigmoid"))

# Compile model
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

# Generate dummy data
import numpy as np
data = np.random.random((1000,100)) # 100 dimensions since the input_dim is 100 dimensions
labels = np.random.randint(2,size=(1000,1)) # Target must be 0D or 1D since the loss is binary

model.summary()

# Train the model
# Fit the model (in general the loss goes down with each epoch)

model.fit(
    data,
    labels,
    batch_size=32,        # Number of examples we train in tandem
    epochs=10,            # Number of times we go through the dataset. Number of samples we go through = number of samples * number of epochs (100*10=1000 samples if data=100)
    verbose=2,            # If 1 shows progress bar, if 2 skips progress bar
    callbacks=None,
    validation_split=0.2, # Split our data such that 0.2 of it is for validation
    validation_data=None, # External source for validation specified
    shuffle=True,         # Always important to shuffle data for algorithms
    class_weight=None,    # Important to specify if weights are not balanced
    sample_weight=None,   # Weight for each sample
    initial_epoch=0)      # Start epoch at a different time

# The model will continue training where it left off
model.train_on_batch(
    data[:32],
    labels[:32],
    class_weight=None,
    sample_weight=None,)

def data_generator():
  for datum, label in zip(data,labels):
    yield datum[None,:], label

# Use the fit_generator() method
model.fit_generator(
    data_generator(),
    steps_per_epoch=900,
    epochs=1,
    verbose=1,
    callbacks=None,
    validation_data=None, # Can be a generator or a dataset
    validation_steps=None,
    class_weight=None,
    max_queue_size=10,
    workers=1,
    initial_epoch=0)

# Evaluation
model.evaluate(
    data,
    labels,
    batch_size=32,
    verbose=1,
    sample_weight=None)

# Prediction
model.predict(
    data,
    batch_size=32,
    verbose=1)
