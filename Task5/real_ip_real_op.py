# Dicision Tree implementation for Real Input(single feature) and Real Output

import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from metrics import *
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

seed_num = 2

class decision_tree_classifier:
    def __init__(self, depth_limit = 10, criteria= "information_gain"):
        self.depth_limit = depth_limit
        self.criteria = criteria      # we are using information gain only

        self.root_node = None
        self.node_num = 0

        self.N = None

    def fit(self, X_train, y_train):
        """
        Inputs :
        X_train : numpy array of shape (num_samples, num_features)
        y_train : numpy array of shape (num_samples, 1)
        """

        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0

        self.N = X_train.shape[0]
        self.P = X_train.shape[1]

        # we are not using the concept of using different attributes along the tree depth
        # we are applying the philosophy of giving the best output irrespective of the fact if some features are left unused

        self.root_node = self.build_tree(X_train, y_train, num=0, parent_node= None)

        return
    
    def build_tree(self, X_temp, y_temp, num, parent_node) :
        """
        Inputs: 
        X_temporary : numpy array of shape (num_samples, num_features)
        y_temporary : numpy array of shape (num_samples, )
        num : node number to be assigned into the node
        parent_node : reference to the parent node

        Outputs:
        node : node object giving the max info gain
        """

        node = Node()
        node.node_num = num
        node.parent = parent_node
        node.samples = X_temp.shape[0]
   
        X_temp= X_temp.reshape(-1,self.P)
        y_temp= y_temp.reshape(-1,1)

        # print('-'*10)
        # print(X_temp)
        # print(y_temp)

        # BASE CASES
        p= node
        current_depth= 0
        while p is not None:
            current_depth += 1
            p= p.parent

        true1= X_temp.shape[0] == 1                  # a single sample is left
        true2= np.std(X_temp) == 0                   # all the samples have the same value (std = 0)       
        true3= current_depth >= self.depth_limit     # when depth limit is reached

        if  true1 or true2 or true3 :
            node.children = {}

            node.mean_decision = np.mean(y_temp)
            node.std = np.std(y_temp)
            node.info_gain = 0
            node.feature = None

            return node


        # Find the best splitting value on the best feature
        info_gain = -np.inf
        sep_val = None
        feature_used = None
        for feature in range(0, self.P):
            lst = X_temp[:, feature]
            lst= list(set(np.sort(lst)))
            for i in range(0,len(lst)-1):   
                possible_sep_val = (lst[i]+lst[i+1])/2
                info_gain_temp = get_info_gain_real_ip_real_op(X_temp, y_temp, possible_sep_val, feature)
                if info_gain_temp > info_gain:
                    info_gain = info_gain_temp 
                    sep_val = possible_sep_val
                    feature_used = feature

        node.info_gain = info_gain
        node.feature = feature_used
        node.sep_val = sep_val
        node.mean_decision = np.mean(y_temp)
        node.std = np.std(y_temp)


        # print('-'*10)
        # print(f"node {node.node_num} X_temp {X_temp}")
        # print(f"node {node.node_num} y_temp {y_temp}")
        # print(f"node {node.node_num} info gain: {info_gain}")
        # print(f"node {node.node_num} sep val: {sep_val}")
        # print(f"node {node.node_num} mean decision: {node.mean_decision}")
        # print(f"node {node.node_num} std: {node.std}")

        # Set the left and right branches
        self.node_num += 1
        X_temp_left = X_temp[X_temp[:,feature_used] <= sep_val]
        y_temp_left = y_temp[X_temp[:,feature_used] <= sep_val]
        node.children["left"] = self.build_tree(X_temp_left, y_temp_left, num = self.node_num, parent_node = node)

        self.node_num += 1
        X_temp_right = X_temp[X_temp[:,feature_used] > sep_val]
        y_temp_right = y_temp[X_temp[:,feature_used] > sep_val]
        node.children["right"] = self.build_tree(X_temp_right, y_temp_right, num = self.node_num, parent_node = node)
        
        return node

    def predict(self, X_test):
        """
        Inputs :
        X_test : numpy array of shape (num_samples, num_features)
        Returns :
        y_hat : numpy array of shape (num_samples, )
        """
        assert self.root_node != None

        y_hat = []
        
        for i in range(X_test.shape[0]):        # sample-wise

            node = self.root_node
            while node.children != {} :
                feat_ = node.feature
                val = X_test[i, feat_]
                
                if val <= node.sep_val:
                    node = node.children["left"]
                else:
                    node = node.children["right"]

            decision = node.mean_decision
            # print(f"X : {X_test[i,0]}    decision :{decision}")
            y_hat.append(decision)

        y_hat = np.array(y_hat).reshape(-1,1)

        return y_hat

    def plot_tree_wrapper(self, graph= None, node= None):
        if graph == None:
            graph = graphviz.Digraph(format= 'png', engine= 'dot') # digraph = Directed graph
            graph.graph_attr['rankdir'] = 'TB'
            graph.graph_attr['nodesep'] = '0.5'

            node = self.root_node

        lab = f"Node Num: {node.node_num}\n"
        lab += f"Samples: {node.samples}\n"
        lab += f"Info Gain: {node.info_gain}\n "
        lab += f"Mean Decision: {node.mean_decision}\n"
        lab += f"Std: {node.std}\n"
        if node.children != {}:
            lab += f"feature {node.feature}: <= {node.sep_val}"

        graph.node(str(id(node)) , label = lab, fillcolor = "lightgreen",style = "filled", shape= 'rectangle')

        if node.children != {} :
            # left branch
            graph.edge(tail_name = str(id(node)), head_name = str(id(node.children["left"])), label= "true")
            graph = self.plot_tree_wrapper(graph, node.children["left"])

            # right branch
            graph.edge(tail_name = str(id(node)), head_name = str(id(node.children["right"])), label= "false")
            graph = self.plot_tree_wrapper(graph, node.children["right"])

        return graph
    
    def plot_tree(self):
        graph = self.plot_tree_wrapper()

        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)

        # Use os.path.join for cross-platform compatibility
        filepath_ = os.path.join('figures', 'realip_realop_dtree')

        graph.render(filename=filepath_, format='png', cleanup=True)

        # Display the graph
        plt.imshow(plt.imread('figures/realip_realop_dtree.png'))
        plt.axis('off')
        plt.figsize = (20,20)
        plt.show()

        # src = graph.source
        # display(graphviz.Source(src, format='png'))

    def plot_dist(self, y_true, X_test, y_test, y_hat):
        """
        Shows the distribution of the true vs predicted data

        Inputs:
        y_true : numpy array of (N, 1) having the original output data without noise
        X_test : numpy array of (N, 1) 
        y_test : numpy array of (N, 1) having the original data with noise
        y_hat : numpy array of (N, 1) having the predicted data
        """

        # plot the y_test
        fig= plt.figure(figsize= (15,5))

        axes = plt.gca()
        axes.plot(X_test[:,0], y_true[:,0], c= "black", label= "Original Test")

        axes.scatter(X_test[:,0], y_test[:,0], c= "blue", s=30, label= "Noisy Test")

        # plot the y_hat
        axes.plot(X_test[:,0], y_hat[:,0], c= "red", label= "Predicted")

        axes.set_xlabel("x-input")
        axes.set_ylabel("y-output")
        axes.set_title("True vs Predicted")

        plt.legend()

        plt.show()

        pass

class Node() :
    def __init__(self):
        self.node_num = None       # int index value
        self.parent = None
        self.children = {}         # will contains references to the left and right branches as {"left":Node, "right":Node}         

        self.info_gain = None
        self.feature = None        # feature used at the node to split
        self.sep_val = None        # contains the separating value
        self.mean_decision = None
        self.std = None            # standard deviation behind the mean decision
        self.samples = None        # number of samples behind the mean decision







# # making the real-ip  dis-op dataset
# N = 100  # num of samples
# P = 1    # num of features

# np.random.seed(seed_num)

# y = np.linspace(1,100, N)
# y = y.reshape(-1,1)
# y_true = 2*y + np.sin(y) + np.cos(y)
# eps = np.random.normal(0, 20, size=(N,1))

# y = y_true + eps

# X = np.linspace(1, 100, N)
# X = X.reshape(-1,1)

# # X_train, y_train, train_true= X, y, y_true
# # X_test= np.random.choice(X.flatten(), size=30).reshape(-1,1)
# # y_test = np.random.choice(y.flatten(), size=30).reshape(-1,1)
# # y_test_true = np.random.choice(y_true.flatten(), size=30).reshape(-1,1)
# X_train, X_test = X, X
# y_train, y_test = y, y
# y_train_true, y_test_true = y_true, y_true


# # calling the decision tree classifier
# dtree = decision_tree_classifier(depth_limit= 5)
# dtree.fit(X_train,y_train)
# y_hat = dtree.predict(X_test)
# dtree.plot_tree()


# # Performance metrics
# rmse = get_rmse(y_hat, y_test)
# print("RMSE: ", rmse)

# mae = get_mae(y_test, y_hat)
# print("MAE: ", mae)

# # plot the distribution plot
# dtree.plot_dist(y_test_true, X_test, y_test, y_hat)

