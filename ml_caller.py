from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve

from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler, Normalizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression as LR

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from pathlib import Path

path = Path.cwd().joinpath('malicious-features').joinpath('features_1k.csv')
data_1 = pd.read_csv(path, encoding = "ISO-8859-1")

# Split the data into input and output variables
x = data_1.iloc[:, 0:22].values   # Input features
y = data_1.iloc[:, 23].values     # Output (capacity)

X_norm = Normalizer().fit_transform(x)

minmaxScaler = MinMaxScaler()
X_minmax = minmaxScaler.fit_transform(x)

X_std = StandardScaler(with_mean=False, with_std=False).fit_transform(x)
X_maxabs = MaxAbsScaler().fit_transform(x)
X_rob = RobustScaler(with_centering=True).fit_transform(x)

std_train_x, std_test_x, std_train_y, std_test_y = train_test_split(X_std, y, test_size=0.2, random_state=42)
minmax_train_x, minmax_test_x, minmax_train_y, minmax_test_y = train_test_split(X_minmax, y, test_size=0.2, random_state=42)
maxabs_train_x, maxabs_test_x, maxabs_train_y, maxabs_test_y = train_test_split(X_maxabs, y, test_size=0.2, random_state=42)
rob_train_x, rob_test_x, rob_train_y, rob_test_y = train_test_split(X_rob, y, test_size=0.2, random_state=42)

norm_train_x, norm_test_x, norm_train_y, norm_test_y = train_test_split(X_norm, y, test_size=0.2, random_state=42)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Set up a cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation
def get_cross_val_score(mdl, x, y):
  scores = cross_val_score(mdl, x, y, cv=kf, scoring='neg_mean_squared_error')
  return np.sqrt(-scores) # convert to RMSE

def print_boxplot(scores,title):
  # Generate a boxplot of the scores
  plt.boxplot(scores)
  plt.title(title + ' Cross-Validation Results')
  plt.xlabel('Model')
  plt.ylabel('RMSE')
  plt.show()
  return

# Random forest model
# Create a random forest regressor model
rf = RandomForestRegressor()
rf.fit(norm_train_x, norm_train_y)
print(rf.score(norm_train_x,norm_train_y))
# Make predictions on the testing set
y_pred_rf = rf.predict(norm_test_x)

# Train the neural network model
#mlp = MLPRegressor(hidden_layer_sizes=(6,12), learning_rate_init=0.00042, max_iter=3000,  random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(22,484), learning_rate_init=0.00032, max_iter=3000, solver='adam', activation='relu', random_state=42)
mlp.fit(norm_train_x, norm_train_y)
print(mlp.score(norm_train_x, norm_train_y))
# Make predictions on the testing set
y_pred_mlp = mlp.predict(norm_test_x)

# fit and evaluate Support Vector Machine model
# 'scale', 'auto'
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
svm = SVR(kernel='poly', coef0=0.75, C=3000, gamma='scale', epsilon=0.1)#0.1, epsilon=.1,max_iter=1000)
svm.fit(norm_train_x, norm_train_y)
svm_score = svm.score(norm_train_x, norm_train_y)
print(svm_score)
# Make predictions on the testing set
y_pred_svm = svm.predict(norm_test_x)

# perform cross-validation and compute MSE for each model
models = [("Support Vector Machine", svm), ("Neural Network", mlp), ("Random Forest", rf)]
mse_results = []

for name, model in models:
    # perform 10-fold cross-validation
    mse_train = -cross_val_score(model, std_train_x, std_train_y, cv=10, scoring='neg_mean_squared_error')
    mse_test = -cross_val_score(model, std_test_x, std_test_y, cv=10, scoring='neg_mean_squared_error')
    
    # store MSE results for each model
    mse_results.append({'model': name, 'mse_train': mse_train, 'mse_test': mse_test})

# plot the boxplots for each model's MSE results
fig, ax = plt.subplots()
bp = ax.boxplot([mse['mse_train'] for mse in mse_results] , vert=False, labels=[mse['model'] for mse in mse_results] )

ax.set_title("Boxplots of MSE for Regression Models Train")
ax.set_xlabel("MSE")
ax.set_ylabel("Model")
plt.show()

fig, ax = plt.subplots()
bp2 = ax.boxplot([mse['mse_test'] for mse in mse_results] , vert=False, labels=[mse['model'] for mse in mse_results] )
ax.set_title("Boxplots of MSE for Regression Models Test")
ax.set_xlabel("MSE")
ax.set_ylabel("Model")
plt.show()