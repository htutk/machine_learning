# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

## Encoding the Independent Variable
## Doesnt quite work
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit(X).toarray()

# use pd.get_dummies()
dummies = np.array(pd.get_dummies(X[:, 0]))
X = np.concatenate((dummies, X[:, 1:]), axis = 1).astype(float)

# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

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