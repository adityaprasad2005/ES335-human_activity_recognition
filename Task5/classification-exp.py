import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from real_ip_discrete_op import decision_tree_classifier as reali_disto_dtree  
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

N, P = X.shape[0], X.shape[1]

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset")
plt.show()

# Ques 2.a) Show the usage of your decision tree on the above dataset. The first 70% of the data should be used
#  for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall
#  of the decision tree you implemented on the test dataset.

# y = y.reshape(-1,1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

# criteria = "information_gain"
# tree = reali_disto_dtree(criteria= criteria)  # Split based on Inf. Gain
# tree.fit(X_train, y_train)
# y_hat = tree.predict(X_test)
# tree.plot_tree()
# print("Criteria :", criteria)
# print("Accuracy: ", get_accuracy(y_hat, y_test))
# for cls in np.unique(y_test):
#     print(f"Precision({cls}): {get_precision(y_hat, y_test, cls)} ")
#     print(f"Recall({cls}): {get_recall(y_hat, y_test, cls)} ")



# Ques 2.b) Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree.

# 5 fold cross-validation
k_outer = 5

fold_size = len(X) // k_outer
depths = [3,4,5,6,7,8]
best_accuracy = -np.inf

for i in range(k_outer):
    depth_ = depths[i]

    X_test = X[i*fold_size : (i+1)*fold_size]
    y_test = y[i*fold_size : (i+1)*fold_size]

    X_train = np.concatenate( (X[0:i*fold_size], X[(i+1)*fold_size: -1]), axis= 0)
    y_train = np.concatenate( (y[0:i*fold_size], y[(i+1)*fold_size: -1]), axis= 0)

    X_test = X_test.reshape(-1, P)
    X_train = X_train.reshape(-1, P)
    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    # calling the decision tree classifier
    dtree = reali_disto_dtree(depth_limit= depth_)
    dtree.fit(X_train, y_train)
    y_hat = dtree.predict(X_test)
    # dtree.plot_tree()
    print(f"------Fold {i} depth {depth_}--------")
    print("Accuracy: ", get_accuracy(y_hat, y_test))
    for cls in np.unique(y_test):
        print(f"Precision({cls}): {get_precision(y_hat, y_test, cls)} ")
        print(f"Recall({cls}): {get_recall(y_hat, y_test, cls)} ")

    
    if get_accuracy(y_hat, y_test) > best_accuracy:
        best_accuracy = get_accuracy(y_hat, y_test)
        best_depth = depth_

print("--------------------")
print(f"optimum depth: {best_depth}")
print(f"Best accuracy: {best_accuracy}")










# Write the code for Q2 a) and b) below. Show your results.

