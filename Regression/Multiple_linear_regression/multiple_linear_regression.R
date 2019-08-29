# Multiple Linear Regression

# importing dataset
dataset = read.csv("50_Startups.csv")
# dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting the multiple linear regression to the training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the training set results
y_pred = predict(regressor, newdata = test_set)

# Backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

# remove state
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

# remove admin
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)
# adj R2 = 0.948 best

# remove marketing spend
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)

# final model
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)