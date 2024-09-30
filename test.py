#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Test functions for the decision tree.
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

from sklearn.metrics import accuracy_score

from tree.tree_node import tree_node
from tree.data_funcs import generate_data


# =============================================================================
# TEST FUCNTIONS
# =============================================================================

def test_method_equivalence():

    data = generate_data()

    train_data = data.loc[data['group'].isin(list(range(5)))]
    train_data = train_data[['X1', 'X2', 'group', 'label']]

    test_data = data.loc[data['group'].isin(list(range(5, 10)))]
    test_data = test_data[['X1', 'X2', 'group', 'label']]


    tree = tree_node(train_data)
    tree.build_tree('gini')

    predictions = tree.evaluate(train_data)

    acc1 = accuracy_score(train_data['label'], predictions)


    tree = tree_node(train_data)
    tree.build_tree('modified_gini', delta=0)

    predictions = tree.evaluate(train_data)

    acc2 = accuracy_score(train_data['label'], predictions)

    assert acc1 == acc2, "The two methods are not equivalent!"



# =============================================================================
# MAIN SCRIPT
# =============================================================================


if __name__ == "__main__":
    
    test_method_equivalence()

    print("All tests passed!")

