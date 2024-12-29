# Dicision Tree implementation for Real Input and Discrete Output

import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from metrics import *
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

seed_num = 0
np.random.seed(seed_num)

class decision_tree_classifier:
    def __init__(self, depth_limit = 10, criteria = "information_gain"):
        """
        Parameters
        ----------
        depth_limit : int
            The maximum depth of the tree.
        criteria : str
            The criteria to use for splitting the tree. Currently the only option is "information_gain".
        """
        
        self.depth_limit = depth_limit
        self.criteria = criteria # we are using information gain only
        self.root_node = None
        self.node_num = 0

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

        # we are using the concept of using attributes multiple times along the tree depth
        # we want to have the most optimum tree
        # we just dont want an attr used consequently
        attr_used = []
        self.root_node = self.build_tree(X_train, y_train, num=0, parent_node= None, attr_used= attr_used)

        return
    
    def build_tree(self, X_temp, y_temp, num, parent_node, attr_used) :
        """
        Inputs: 
        X_temporary : numpy array of shape (num_samples, num_features)
        y_temporary : numpy array of shape (num_samples, )
        num : node number to be assigned into the node
        parent_node : reference to the parent node
        attr_used : list of features that have already been used

        Outputs:
        node : node object giving the max info gain
        """

        # print('-'*10)
        # print(X_temp)
        # print(y_temp)
        node = Node()
        node.node_num = num
        node.parent = parent_node
   
        # print("attr_used: ", attr_used)

        # BASE CASES
        if len(np.unique(y_temp)) == 1: # all the samples have the same label
            node.feature = None
            node.children = {}
            node.majority_decision = y_temp[0,0]
            node.probability = 1
            node.info_gain = 0

            return node

        true1= len(attr_used) == X_temp.shape[1]     # when all the features have been used
        # true2= len(attr_used) >= self.depth_limit    # when depth limit is reached
        true2= np.log2(self.node_num+ 1e-3) >= self.depth_limit # when depth limit is reached

        if  true1 or true2 :
            node.feature = None
            node.children = {}

            vals, counts = np.unique(y_temp, return_counts=True)
            node.majority_decision = vals[counts.argmax()]
            node.probability = max(counts)/counts.sum()
            node.info_gain = 0

            return node

        # we choose any random attr among the unused attributes
        attr = np.random.choice(np.setdiff1d(np.arange(self.P), attr_used)) 
        node.feature = attr

        vals, counts = np.unique(y_temp, return_counts=True)
        node.majority_decision = vals[counts.argmax()]
        node.probability = max(counts)/counts.sum()

        # Find the best splitting value
        lst = X_temp[:, attr]
        lst = list(set(np.sort(lst)))
        info_gain = -np.inf
        sep_val = None
        for i in range(0,len(lst)-1):   
            possible_sep_val = (lst[i]+lst[i+1])/2
            # print("possible sep val", possible_sep_val)
            info_gain_temp = get_info_gain_real_ip_dis_op(X_temp, y_temp, attr, possible_sep_val)
            if info_gain_temp > info_gain:
                info_gain = info_gain_temp 
                sep_val = possible_sep_val

        node.info_gain = info_gain
        node.sep_val = sep_val

        # we want to use an attribute multiple times in the flow of the tree
        # attr_used_ = attr_used + [attr]
        attr_used_ = [attr]

        # Set the left and right branches
        self.node_num += 1
        X_temp_left = X_temp[X_temp[:,attr] <= sep_val]
        y_temp_left = y_temp[X_temp[:,attr] <= sep_val]
        node.children["left"] = self.build_tree(X_temp_left, y_temp_left, num = self.node_num, parent_node = node, attr_used= attr_used_ )

        self.node_num += 1
        X_temp_right = X_temp[X_temp[:,attr] > sep_val]
        y_temp_right = y_temp[X_temp[:,attr] > sep_val]
        node.children["right"] = self.build_tree(X_temp_right, y_temp_right, num = self.node_num, parent_node = node, attr_used= attr_used_)
        
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
                
                if val < node.sep_val:
                    node = node.children["left"]
                else:
                    node = node.children["right"]

            decision = node.majority_decision

            y_hat.append(decision)

        y_hat = np.array(y_hat)
        y_hat = y_hat.reshape(-1,1)

        return y_hat

    def plot_tree_wrapper(self, graph= None, node= None):
        if graph == None:
            graph = graphviz.Digraph(format= 'png', engine= 'dot') # digraph = Directed graph
            graph.graph_attr['rankdir'] = 'TB'
            graph.graph_attr['nodesep'] = '0.5'

            node = self.root_node

        lab = f"Node Num: {node.node_num}\n"
        lab += f"Info Gain: {node.info_gain}\n "
        lab += f"Majority Decision: {node.majority_decision}\n"
        lab += f"Probability: {node.probability}\n"

        if node.feature != None:
            lab += f"Feature {node.feature} <= {node.sep_val}"

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
        filepath_ = os.path.join('figures', 'realip_distop_dtree')

        graph.render(filename=filepath_ , format='png', cleanup=True)

        # Display the graph
        plt.imshow(plt.imread('figures/realip_distop_dtree.png'))
        plt.axis('off')
        plt.figsize = (20,20)
        plt.show()

        # src = graph.source
        # display(graphviz.Source(src, format='png'))

    def plot_dist(self, X_test, y_test, y_hat):
        """
        Shows the distribution of the data classified by the binary decision tree
        """

        # plot the y_test
        fig , ax = plt.subplots(figsize=(5,10))

        classes = np.unique(y_test)
        color = ["blue" if i==classes[0] else "red" for i in y_test]
        print("blue : ", classes[0])
        print("red : ", classes[1])

        ax.scatter(X_test[:,0], X_test[:,1], c= color, s=30)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        
        # make the meshgrid and plot the decision boundary
        x1_lim = ax.get_xlim()
        x2_lim = ax.get_ylim()

        x1 = np.linspace(x1_lim[0], x1_lim[1], 100)
        x2 = np.linspace(x2_lim[0], x2_lim[1], 100)

        xx1, xx2 = np.meshgrid(x1, x2)
        Z = self.predict(np.c_[xx1.flatten(), xx2.flatten()]).reshape(xx1.shape)

        # plot the decision boundary
        # ax.contourf(xx1, xx2, Z, cmap='winter', alpha=0.3)
        color= []
        for i in Z.flatten():
            if i == classes[0]:
                color.append("blue")
            else:
                color.append("red")
        ax.scatter(xx1.flatten(), xx2.flatten(), c= color, alpha =0.5, s=1 )
        ax.set_title("Decision Boundary")

        plt.show()

        pass

class Node() :
    def __init__(self):
        self.node_num = None       # int index value
        self.feature = None        # the feature about which it splits the data
        self.parent = None
        self.children = {}         # will contains references to the left and right branches as {"left":Node, "right":Node}         

        self.info_gain = None
        self.sep_val = None        # contains the separating value
        self.majority_decision = None
        self.probability = None    # probability behind the majority decision





# making the real-ip  dis-op dataset
# N = 500  # num of samples
# P = 2    # num of features = 2 otherwise dtree.plot_dist() won't work
# C = 3    # num of classes of output variable

# lst_vals = [str(i) for i in range(C)]
# y = [np.random.choice(lst_vals) for i in range(N)]
# X = np.random.randint(1,100,size=(N,P))


# # from sklearn.datasets import make_blobs
# # X, y = make_blobs(n_samples=N, centers=4,
# #                   random_state=seed_num, cluster_std=1.2)
# # y= [str(i) for i in y]

# y = np.array(y)
# y = y.reshape(-1,1)
# X = np.array(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed_num)
# # print(X_train)
# # print(y_train)

# # calling the decision tree classifier
# dtree = decision_tree_classifier(depth_limit= 4)
# dtree.fit(X_train,y_train)
# y_hat = dtree.predict(X_test)
# dtree.plot_tree()

# # plot the distribution plot
# dtree.plot_dist(X_test, y_test, y_hat)



# # Performance metrics
# # print("Accuracy: ", get_accuracy(y_hat, y_test))
# # for cls in np.unique(y):
# #     print('Class:', cls)
# #     print("Precision: ", get_precision(y_hat, y_test, cls))
# #     print("Recall: ", get_recall(y_hat, y_test, cls))
# #     print("F1 Score: ", get_f1_score(y_hat, y_test, cls))

# # cm = confusion_matrix(y_hat, y_test)
# # print(cm)
# # print("Matthews Coefficient: ", get_mcc_score(y_hat, y_test))




