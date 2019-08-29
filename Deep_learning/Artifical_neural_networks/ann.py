# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.11/get_started

# Installing Keras
# pip install --upgrade keras

# updated version:
# conda install -c conda-forge keras (on anaconda prompt)

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
length = dataset.shape[1]
# Exclude RowNum, CustomerId, Surname
X = dataset.iloc[:, 3:length - 1].values
y = dataset.iloc[:, length - 1].values

# Creating dummy variables
geo_dummy = pd.get_dummies(X[:, 1])
gender_dummy = pd.get_dummies(X[:, 2])
remove_index = [i for i in range(X.shape[1])]
remove_index.pop(1)  # Remove geography col
remove_index.pop(1)  # Remove gender col
X = X[:, remove_index]

# Avoiding the dummy variable trap
geo_dummy = geo_dummy.iloc[:, 1:3].values
gender_dummy = gender_dummy.iloc[:, 1:2].values

# Concatenate the X
X = np.concatenate((geo_dummy, X[:, :]), axis = 1).astype(float)
X = np.concatenate((X[:, :3], gender_dummy, X[:, 3:]), axis = 1).astype(float)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
# test_size is a small percentage = 20% - 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Part 2: Making ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the first layer: input layer and the first hidden layer
# missing input_shape parameter
classifier.add(layer = Dense(units = 128, activation = "relu"))

# Add the second layer
classifier.add(layer = Dense(units = 128, activation = "relu"))

# Add the output layer
classifier.add(layer = Dense(units = 1, activation = "sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting ANN to the training stet
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3: Making predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)