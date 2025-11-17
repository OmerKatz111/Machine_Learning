import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    classes = data[:, -1]
    _, classesCount = np.unique(classes, return_counts=True)
    classesProbs = classesCount / len(classes)
    gini = 1 - sum(classesProbs ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    classes = data[:, -1]
    _, classesCount = np.unique(classes, return_counts=True)
    classesProbs = classesCount / len(classes)
    classesLogProbs = np.log2(classesProbs)
    entropy = -1 * sum(classesProbs * classesLogProbs)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        classesCol = self.data[:, -1]
        classes, classesCount = np.unique(classesCol, return_counts=True)
        pred = classes[np.argmax(classesCount)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        feature_values = self.data[:, feature]
        unique_values = np.unique(feature_values)

        for val in unique_values:
            groups[val] = self.data[feature_values == val]

        total_size = len(self.data)
        impurity_before = self.impurity_func(self.data)

        impurity_after = 0
        for val, group_data in groups.items():
            impurity_after += (len(group_data) / total_size) * self.impurity_func(group_data)

        goodness = impurity_before - impurity_after

        if self.gain_ratio:
            impurity_before = calc_entropy(self.data)
            impurity_after = 0
            for val, group_data in groups.items():
                impurity_after += (len(group_data) / total_size) * calc_entropy(group_data)
            info_gain = impurity_before - impurity_after

            probs = np.array([len(group_data) / total_size for group_data in groups.values()])
            split_info = -np.sum(probs * np.log2(probs + 1e-10))

            if split_info == 0:
                goodness = 0
            else:
                goodness = info_gain / split_info
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.feature == -1:
            self.feature_importance = 0
        else:
            nodeProb = len(self.data) / n_total_sample
            goodness, _ = self.goodness_of_split(self.feature)
            self.feature_importance = nodeProb * goodness
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ##########################################################################
        #TODO: Implement the function.                                           #
        ##########################################################################
        if self.impurity_func(self.data) == 0:
            self.terminal = True
        if self.terminal or self.depth >= self.max_depth:
            return

        best_feature = None
        best_score = -np.inf
        best_groups = {}

        for featureIndex in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(featureIndex)
            if goodness > best_score:
                best_score = goodness
                best_feature = featureIndex
                best_groups = groups

        if best_feature is None or best_score <= 0:
            self.terminal = True
            return

        if self.chi < 1:
            labels = np.unique(self.data[:, -1])
            feature_values = np.unique(self.data[:, best_feature])

            if len(feature_values) <= 1 or len(labels) <= 1:
                pass
            else:
                contingency = np.zeros((len(feature_values), len(labels)))

                feature_value_to_index = {val: idx for idx, val in enumerate(feature_values)}
                label_to_index = {label: idx for idx, label in enumerate(labels)}

                for val, group in best_groups.items():
                    if val in feature_value_to_index:
                        for label in labels:
                            if label in label_to_index:
                                count = np.sum(group[:, -1] == label)
                                contingency[feature_value_to_index[val], label_to_index[label]] = count

                row_sums = contingency.sum(axis=1)
                col_sums = contingency.sum(axis=0)
                total = contingency.sum()

                if total > 0:
                    expected = np.outer(row_sums, col_sums) / total

                    mask = expected > 0
                    if expected[mask].shape[0] > 0:
                        chi_squared_stat = (((contingency - expected)[mask] ** 2) / expected[mask]).sum()
                    else:
                        chi_squared_stat = 0

                    dof = (len(feature_values) - 1) * (len(labels) - 1)

                    if dof > 0:
                        threshold = get_chi_square_threshold(dof, self.chi)
                        if chi_squared_stat < threshold:
                            self.terminal = True
                            return

        self.feature = best_feature

        for feature_value, subset in best_groups.items():
            if subset.shape[0] == 0:
                continue
            child_node = DecisionNode(
                data=subset,
                impurity_func=self.impurity_func,
                feature=-1,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child_node, feature_value)
        ##########################################################################
        #                            END OF YOUR CODE                            #
        ##########################################################################



class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #

    def depth(self):
        return self.root.depth

    def split_recursive(self, node):
        node.split()
        for child in node.children:
            if not child.terminal:
                self.split_recursive(child)


    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            feature=-1,
            depth=0,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )

        self.split_recursive(self.root)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        while not node.terminal:
            feature_value = instance[node.feature]
            found = False
            for i, val in enumerate(node.children_values):
                if val == feature_value:
                    node = node.children[i]
                    found = True
                    break
            if not found:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        correct = 0

        for instance in dataset:
            pred = self.predict(instance)
            true_label = instance[-1]
            if pred == true_label:
                correct += 1

        accuracy = (correct / len(dataset)) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            chi=1,
            max_depth=max_depth,
            gain_ratio=True
        )
        tree.build_tree()

        train_acc = tree.calc_accuracy(X_train)
        val_acc = tree.calc_accuracy(X_validation)

        training.append(train_acc)
        validation.append(val_acc)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for chi_val in chi_values:
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            chi=chi_val,
            max_depth=1000,
            gain_ratio=True
        )
        tree.build_tree()

        train_acc = tree.calc_accuracy(X_train)
        val_acc = tree.calc_accuracy(X_test)
        chi_training_acc.append(train_acc)
        chi_validation_acc.append(val_acc)
        tree_depth = calc_depth(tree.root)
        depth.append(tree_depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.
    ###########################################################################
    if node.terminal:
        return 1
    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes

def get_chi_square_threshold(dof, chi):
    dof = min(dof, max(chi_table.keys()))
    return chi_table[dof][chi]

def calc_depth(node):
    if node.terminal or not node.children:
        return node.depth

    depths = []
    for child in node.children:
        depths.append(calc_depth(child))

    return max(depths)