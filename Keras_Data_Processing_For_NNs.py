import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Create an empty array for the training samples and labels
train_samples = []
train_labels = []

# Experiment
# Drug tested on 2100 people ages 13-100.
# 50% over 65 and 50% under 65.
# 95% 65+ had side effects and 95% below 65 did not

for i in range(50):
    # The 5% 13-64 who experienced side effects
    random_younger = randint(13,64)             # Generate random number between 13, 64
    train_samples.append(random_younger)        # Append the random number to the train_samples array
    train_labels.append(1)                      # Append a 1 to trained_labels array to indicate the patient had side effects

    # The 5% 65-100 who did not have side effects
    random_older = randint(65,100)              # Generate random number between 65, 100
    train_samples.append(random_older)          # Append the random number to the train_samples array
    train_labels.append(0)                      # Append a 0 to trained_labels array to indicate they did not have side effects

for i in range(1000):
    # The 95% younger who did not have side effects
    random_younger = randint(13, 64)            # Generate random number between 13, 64
    train_samples.append(random_younger)        # Append the random number to the train_samples array
    train_labels.append(0)                      # Append a 0 to trained_labels array to indicate they did not have side effects

    # The 95% older who did have side effects
    random_older = randint(65, 100)             # Generate random number between 65, 100
    train_samples.append(random_older)          # Append the random number to the train_samples array
    train_labels.append(1)                      # Append a 1 to trained_labels array to indicate they did have side effects

# Print the values added to the trained_samples array
for i in train_samples:
   print(i)

# Print the values added to the trained_labels array (all 0 and 1)
for i in train_labels:
    print(i)

# Convert arrays into numpy arrays
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)

train_samples, train_labels = shuffle(train_samples,train_labels)         # Randomly shuffle the values in the arrays

scaler = MinMaxScaler(feature_range=(0,1))                                # Rescale our range from 13-100 to 0-1 and fit_transform does not do 1D data so needed to reshape it
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))  # Fit to data by computing the min and max values and transform it 

for i in scaled_train_samples:
    print(i)
