# Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into training set and test set
#from sklearn.model_selection import train_test_split
## test_size is a small percentage = 20% - 40%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
#                                                    random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result with Random Forest Regression
y_pred = regressor.predict([[6.5]])

# Visualizing the Random Forest Regression results (for higher res and smoother curve)
plt.figure(1)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


plt.figure(2)
x_ln = np.arange(0, 1, 0.0001)
y_ln = np.log(x_ln / (1 - x_ln))
plt.plot(x_ln, y_ln, color = 'blue')
plt.xlim(-1, 5)
plt.ylim(-10, 10)
plt.show()
