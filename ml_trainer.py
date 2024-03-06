from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve

from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler, Normalizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
#from sklearn.linear_model import LogisticRegression as LR

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from pathlib import Path

import pickle

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

curX =    rob_train_x
curY =    rob_train_y
curTestX =  rob_test_x
curTestY =  rob_test_y

# CTree
clf = tree.DecisionTreeClassifier(criterion='entropy')  #'entropy' or 'gini'
clf.fit(train_x,train_y)
print(clf.score(train_x,train_y))
y_pred_clf = clf.predict(test_x)

pickle.dump(clf, open('clf_mdl', "wb"))

# Random forest model
# Create a random forest regressor model
rf = RandomForestRegressor()
rf.fit(curX, curY)
print(rf.score(curX,curY))
# Make predictions on the testing set
y_pred_rf = rf.predict(curTestX)

pickle.dump(rf, open('rf_mdl', "wb"))

# Train the neural network model
#mlp = MLPRegressor(hidden_layer_sizes=(6,12), learning_rate_init=0.00042, max_iter=3000,  random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(22,154), learning_rate_init=0.00032, max_iter=3000, solver='adam', activation='relu', random_state=42)
mlp.fit(curX, curY)
print(mlp.score(curX, curY))
# Make predictions on the testing set
y_pred_mlp = mlp.predict(curTestX)

pickle.dump(mlp, open('mlp_mdl', "wb"))

# fit and evaluate Support Vector Machine model
# 'scale', 'auto'
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
svmX = norm_train_x
svmY = norm_train_y
svmTestX =  norm_test_x
svmTestY =  norm_test_y

svm = SVR(kernel='poly', coef0=0.75, C=3000, gamma='scale', epsilon=0.1)#0.1, epsilon=.1,max_iter=1000)
svm.fit(svmX, svmY)
svm_score = svm.score(svmX, svmY)
print(svm_score)
# Make predictions on the testing set
y_pred_svm = svm.predict(svmTestX)

pickle.dump(svm, open('svm.mdl', "wb"))

# C Tree
print("---------------------------CTree-MODE---------------------------")
scores = get_cross_val_score(clf, c_x_train, c_y_train)
print_boxplot(scores, 'CTree')
mse = mean_squared_error(c_y_test, y_pred_clf)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(c_y_test, y_pred_clf)
print("Mean squared error (MSE): {:.2f}".format(mse))
print("Root mean squared error (RMSE): {:.2f}".format(rmse))
print("Coefficient of determination (R-squared): {:.2f}".format(r2))
print("--------------------------------------------")

# Random Forest
print("---------------------------RF-MODE---------------------------")
scores = get_cross_val_score(rf, curX, curY)
print_boxplot(scores, 'RF')
mse = mean_squared_error(curTestY, y_pred_rf)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(curTestY, y_pred_rf)
print("Mean squared error (MSE): {:.2f}".format(mse))
print("Root mean squared error (RMSE): {:.2f}".format(rmse))
print("Coefficient of determination (R-squared): {:.2f}".format(r2))
print("--------------------------------------------")

# MLP
print("---------------------------MLP-MODE--------------------------")
scores = get_cross_val_score(mlp, curX, curY)
print_boxplot(scores, 'MLP')
mse = mean_squared_error(curTestY, y_pred_mlp)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(curTestY, y_pred_mlp)
print("Mean squared error (MSE): {:.2f}".format(mse))
print("Root mean squared error (RMSE): {:.2f}".format(rmse))
print("Coefficient of determination (R-squared): {:.2f}".format(r2))
print("--------------------------------------------")

# SVM
print("---------------------------SVM-MODE--------------------------")
scores = get_cross_val_score(svm, svmX, svmY)
print_boxplot(scores, 'SVM')
mse = mean_squared_error(svmTestY, y_pred_svm)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(svmTestY, y_pred_svm)
print("Mean squared error (MSE): {:.2f}".format(mse))
print("Root mean squared error (RMSE): {:.2f}".format(rmse))
print("Coefficient of determination (R-squared): {:.2f}".format(r2))
print("--------------------------------------------")

