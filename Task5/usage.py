"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output

# N = 30 
# P = 5
# X = np.random.randn(N, P)
# y = np.random.randn(N, 1)

# from real_ip_real_op import decision_tree_classifier as reali_realo_dtree # type:ignore

# for criteria in ["information_gain"]:
#     tree = reali_realo_dtree(criteria=criteria) 
#     tree.fit(X, y)
#     y_hat = tree.predict(X)
#     tree.plot_tree()
#     print("Criteria :", criteria)
#     print("RMSE: ", get_rmse(y_hat, y))
#     print("MAE: ", get_mae(y_hat, y))

# Test case 2
# Real Input and Discrete Output

# N = 30
# P = 5 # num of classes
# X = np.random.randn(N, P)

# y = [str(i) for i in np.random.randint(P, size=N)]
# y = np.array(y).reshape(-1,1)

# from real_ip_discrete_op import decision_tree_classifier as reali_disto_dtree # type:ignore

# for criteria in ["information_gain"]:
#     tree = reali_disto_dtree(criteria= criteria)  # Split based on Inf. Gain
#     tree.fit(X, y)
#     y_hat = tree.predict(X)
#     tree.plot_tree()
#     print("Criteria :", criteria)
#     print("Accuracy: ", get_accuracy(y_hat, y))
#     for cls in np.unique(y):
#         print(f"Precision({cls}): {get_precision(y_hat, y, cls)} ")
#         print(f"Recall({cls}): {get_recall(y_hat, y, cls)} ")


# Test case 3
# Discrete Input and Discrete Output

# N = 30
# P = 4

# y = [np.random.choice([ "no", "yes"]) for i in range(N)]
# f1_cls = [ "sunny", "overcast", "rain"]
# f2_cls = [ "hot", "mild", "cold"]
# f3_cls = [ "high", "normal"]
# f4_cls = [ "weak", "strong"]
# X = [[np.random.choice(f1_cls) , np.random.choice(f2_cls) , np.random.choice(f3_cls), np.random.choice(f4_cls)]for i in range(N)]

# X = np.array(X).reshape(-1,P)
# y = np.array(y).reshape(-1,1)


# from discrete_ip_discrete_op import decision_tree_classifier as disti_disto_dtree # type:ignore

# for criteria in ["information_gain"]:
#     tree = disti_disto_dtree(criteria= criteria)  # Split based on Inf. Gain
#     tree.fit(X, y)
#     y_hat = tree.predict(X)
#     tree.plot_tree()
#     print("Criteria :", criteria)
#     print("Accuracy: ", get_accuracy(y_hat, y))
#     for cls in np.unique(y):
#         print(f"Precision({cls}): {get_precision(y_hat, y, cls)}" )
#         print(f"Recall({cls}): {get_recall(y_hat, y, cls)}" )

# Test case 4
# Discrete Input and Real Output

N = 30
P = 5

y = np.random.randint(1,100,size= N)
f1_cls = [ "sunny", "overcast", "rain"]
f2_cls = [ "hot", "mild", "cold"]
f3_cls = [ "high", "normal"]
f4_cls = [ "weak", "strong"]
X = [[np.random.choice(f1_cls) , np.random.choice(f2_cls) , np.random.choice(f3_cls), np.random.choice(f4_cls)]for i in range(N)]

y = np.array(y)
y = y.reshape(-1,1)
X = np.array(X)

from discrete_ip_real_op import decision_tree_classifier as disti_realo_dtree # type:ignore

for criteria in ["information_gain"]:
    tree = disti_realo_dtree(criteria= criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot_tree()
    print("Criteria :", criteria)
    print("RMSE: ", get_rmse(y_hat, y))
    print("MAE: ", get_mae(y_hat, y))
