#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python module containing necessary functions to build a decision tree.
"""


# =============================================================================
# AUTHOR INFORMATION
# =============================================================================

__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2024, McCoy and Peterson"
__credits__ = ["Kevin McCoy", "Christine Peterson"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Christine Peterson"
__email__ = ["CBPeterson@mdanderson.org", "kmccoy1@rice.edu"]
__status__ = "development"
__date__ = "2024-02-08" # Last modified date

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import math
from tree.cost_funcs import gini_index, group_gini_index, sum_of_squares
from tree.data_funcs import plot_data


# =============================================================================
# tree_node class
# =============================================================================



class tree_node():
    """Base class for a decision tree node.
    """

    def __init__(self, data, depth=0):
        """_summary_

        Args:
            data (_type_): _description_
            depth (int, optional): _description_. Defaults to 0.
        """

        self.data = data
        self.depth = depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.label = None

    def set_leaf(self, label):
        """_summary_

        Args:
            label (_type_): _description_
        """

        self.is_leaf = True
        self.label = label

    def set_split(self, split_feature, split_value):
        """_summary_

        Args:
            split_feature (_type_): _description_
            split_value (_type_): _description_
        """

        self.split_feature = split_feature
        self.split_value = split_value

    def set_left(self, left):
        """_summary_

        Args:
            left (_type_): _description_
        """

        self.left = left

    def set_right(self, right):
        """_summary_

        Args:
            right (_type_): _description_
        """

        self.right = right



    def predict(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.is_leaf == True:
            return self.label
        else:
            if (data[self.split_feature] <= self.split_value).any():
                return self.left.predict(data)
            else:
                return self.right.predict(data)
            

    
    def print_tree(self):
        """_summary_
        """

        my_dict = {'split': self.split_feature, 
                    'value': self.split_value.item(), 
                    'left': self.left.print_tree() if self.left.label is None else int(self.left.label),
                    'right': self.right.print_tree() if self.right.label is None else int(self.right.label)
                   }

        return my_dict




    def evaluate(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        predictions = []
        for i in range(len(data)):
            predictions.append(self.predict(data.iloc[i]))
        return predictions
    


    def build_tree(self, method, max_depth=10, min_samples_split=2, min_samples_leaf=1, delta=0.5):
        """_summary_

        Args:
            tree (_type_): _description_
            max_depth (int, optional): _description_. Defaults to 5.
            min_samples_split (int, optional): _description_. Defaults to 2.
            min_samples_leaf (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """

        # Test terminating conditions

        if len(self.data['label'].unique()) == 1:
            self.set_leaf(self.data['label'].mode()[0])
            return
        
        if len(self.data) <= min_samples_split:
            self.set_leaf(self.data['label'].mode()[0])
            return
        
        if len(self.data) <= min_samples_leaf:
            self.set_leaf(self.data['label'].mode()[0])
            return
        
        if self.depth == max_depth:
            self.set_leaf(self.data['label'].mode()[0])
            return


        # find the best split
        best_split_feature = None
        best_split_value = None
        best_split_gini = math.inf
        for feature in self.data.columns[:-2]:

            sorted_ = np.sort(self.data[feature].unique())
            for value in sorted_[:-1]:
                left_data = self.data[self.data[feature] <= value]
                right_data = self.data[self.data[feature] > value]

                if method == 'gini':
                    left_gini = gini_index(left_data)
                    right_gini = gini_index(right_data)
                elif method == 'modified_gini':
                    left_gini = delta*group_gini_index(left_data) + (1-delta)*gini_index(left_data)
                    right_gini = delta*group_gini_index(right_data) + (1-delta)*gini_index(right_data)

                # TODO: Implement sum of squares cost function
                # TODO: Implement ICC idea

                gini_value = (len(left_data) * left_gini + len(right_data) * right_gini) / len(self.data)
                if gini_value < best_split_gini:
                    best_split_feature = feature
                    best_split_value = value
                    best_split_gini = gini_value

        # split the data
        left_data = self.data[self.data[best_split_feature] <= best_split_value]
        right_data = self.data[self.data[best_split_feature] > best_split_value]

        # build the tree
        self.set_split(best_split_feature, best_split_value)
        self.set_left(tree_node(left_data, self.depth + 1))
        self.set_right(tree_node(right_data, self.depth + 1))

        self.left.build_tree(method, delta=delta)
        self.right.build_tree(method, delta=delta)



    def plot_tree(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError
        # TODO: Implement a function to plot the tree

        

    def prune_tree(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError
    
        # TODO: Implement a function to prune the tree

        

    

