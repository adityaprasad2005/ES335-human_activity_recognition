# Dicision Tree implementation for Discrete Input and Discrete Output

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
    def __init__(self, depth_limit = 10, criteria = "information_gain"):
        self.depth_limit = depth_limit
        self.criteria = criteria                # we are using information gain(entropy) only
        self.root_node = None
        self.node_num = 0

        self.N = None
        self.P = None

    def fit(self, X_train, y_train):
        """
        Inputs :
        X_train : numpy array of shape (num_samples, num_features)
        y_train : numpy array of shape (num_samples, )
        """

        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0

        self.N = X_train.shape[0]
        self.P = X_train.shape[1]

        self.root_node = self.build_tree(X_train, y_train, num=0, parent_node= None)

        return
    
    def build_tree(self, X_temp, y_temp, num, parent_node) :
        """
        Inputs: 
        X_temporary : numpy array of shape (num_samples, num_features)
        y_temporary : numpy array of shape (num_samples, )

        Outputs:
        node : node object giving the max info gain
        """

        # print('-'*10)
        # print(X_temp)
        # print(y_temp)
        node = Node()
        node.node_num = num
        node.parent = parent_node

        attr_used = []
        p = parent_node
        while p is not None:
            attr_used.append(p.feature)
            p = p.parent
   
        # print("attr_used: ", attr_used)

        # BASE CASES
        if len(np.unique(y_temp)) == 1: # all the samples have the same label
            node.feature = None
            node.children = {}
            node.majority_decision = y_temp[0]
            node.probability = 1
            node.info_gain = 0

            return node

        true1= len(attr_used) == X_temp.shape[1]     # when all the features have been used
        true2= len(attr_used) >= self.depth_limit    # when depth limit is reached
        true3= (X_temp.shape[1] == 1+ len(attr_used)) & ( len(np.unique(X_temp[:, np.setdiff1d(np.arange(self.P),attr_used) ])) ==1)    # when all except one feature is left for use and that column in X has all the same classes

        if  true1 or true2 or true3:
            node.feature = None
            node.children = {}

            vals, counts = np.unique(y_temp, return_counts=True)
            node.majority_decision = vals[counts.argmax()]
            node.probability = max(counts)/counts.sum()
            node.info_gain = 0

            return node

        # find the attribute with the max info gain 
        info_gain = -np.inf
        attr = None
        for attr_num in range(X_temp.shape[1]):

            if attr_num in attr_used:        # skip if that attribute has already been used
                continue

            info_gain_temp = get_info_gain_dis_ip_dis_op(X_temp, y_temp, attr_num)
            # print(f"Info Gain of attribute {attr_num} : {info_gain_temp:4f}")

            if info_gain_temp > info_gain:
                info_gain = info_gain_temp
                attr = attr_num
        # print("max info gain :", info_gain)
        # print("attribute :", attr)

        node.feature = attr
        node.info_gain = info_gain
        vals, counts = np.unique(y_temp, return_counts=True)
        node.majority_decision = vals[counts.argmax()]
        node.probability = max(counts)/counts.sum()

        # RECURSIVE CASES
        unique_vals = np.unique(X_temp[:, attr])
        for val in unique_vals:
            # print("val: ", val)
            X_temp_temp = X_temp[X_temp[:, attr] == val]
            # X_temp_temp = np.delete(X_temp_temp, attr, axis = 1)
            y_temp_temp = y_temp[X_temp[:, attr] == val]

            self.node_num += 1
            node.children[val] = self.build_tree(X_temp_temp, y_temp_temp, num = self.node_num, parent_node = node)

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
                attr_num = node.feature
                val = X_test[i, attr_num]
                try:
                    node = node.children[val]
                    bool_val = False
                except:
                    bool_val = True
                    break

            if bool_val:
                decision = {} # stores the decision and its prob as {"Yes":0.91, "No":0.09}
                for other_cls, other_child in node.children.items():
                    if decision[other_child.majority_decision] :
                        decision[other_child.majority_decision] += other_child.probability
                    else:
                        decision[other_child.majority_decision] = other_child.probability

                keys, vals = decision.items()
                decision = keys[np.argmax(vals)]

            else:
                decision = node.majority_decision

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
        lab += f"feature : {node.feature}\n"
        lab += f"Info Gain: {node.info_gain}\n "
        lab += f"Majority Decision: {node.majority_decision}\n"
        lab += f"Probability: {node.probability}"

        graph.node(str(id(node)) , label = lab, fillcolor = "lightgreen",style = "filled", shape= 'rectangle')

        for sep_class, child_node in node.children.items():
            graph.edge(tail_name = str(id(node)), head_name = str(id(child_node)), label= sep_class)
            graph = self.plot_tree_wrapper(graph, child_node)

        return graph
    
    def plot_tree(self):
        graph = self.plot_tree_wrapper()

        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)

        # Use os.path.join for cross-platform compatibility
        filepath_ = os.path.join('figures', 'disip_disop_dtree')

        graph.render(filename=filepath_, format='png', cleanup=True)

        # Display the graph
        plt.imshow(plt.imread('figures/disip_disop_dtree.png'))
        plt.axis('off')
        plt.figsize = (20,20)
        plt.show()

        # src = graph.source
        # display(graphviz.Source(src, format='png'))

class Node() :
    def __init__(self):
        
        self.node_num = None       # int index value
        self.feature = None        # the feature about which it splits the data
        self.parent = None
        self.children = {}         # will contains references to the child nodes in the form of key:val pairs as {"rainy":Node, } 

        self.info_gain = None
        self.majority_decision = None
        self.probability = None    # probability behind the majority decision







# making the dis-ip  dis-op dataset
# N = 5  # num of samples
# P = 4   # num of features

# np.random.seed(seed_num)
# y = [np.random.choice([ "no", "yes"]) for i in range(N)]
# X = [[np.random.choice([ "sunny", "overcast", "rain"]) , np.random.choice([ "hot", "mild", "cold"]) , np.random.choice([ "high", "normal"]), np.random.choice([ "weak", "strong"])]for i in range(N)]

# y = np.array(y)
# y = y.reshape(-1,1)
# X = np.array(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# # print(X_train)
# # print(y_train)

# # calling the decision tree classifier
# dtree = decision_tree_classifier()
# dtree.fit(X_train,y_train)
# y_hat = dtree.predict(X_test)
# dtree.plot_tree()

# # Perfermonce metrics
# print("Accuracy: ", get_accuracy(y_hat, y_test))
# for cls in np.unique(y):
#     print('Class:', cls)
#     print("Precision: ", get_precision(y_hat, y_test, cls))
#     print("Recall: ", get_recall(y_hat, y_test, cls))
#     print("F1 Score: ", get_f1_score(y_hat, y_test, cls))

# cm = confusion_matrix(y_hat, y_test)
# print(cm)
# print("Methews Coefficient: ", get_mcc_score(y_hat, y_test))
