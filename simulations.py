#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script to run multiple experiments in parallel using the 
multiprocessing library.
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


# Standard Library
from multiprocessing import Pool
from tqdm import tqdm

# Import local functions
from run_experiment import run_experiment


# =============================================================================
# SCRIPT
# =============================================================================


if __name__ == '__main__': 

    # Set number of iterations and processes
    NUM_ITER = 50
    NUM_PROCESSES = 6

    # Run experiments in parallel
    pool = Pool(NUM_PROCESSES)

    # Run with progress bar
    for _ in tqdm(pool.imap_unordered(run_experiment, list(range(NUM_ITER))),
                        total=NUM_ITER, position=0, leave=True):
        pass

    # Close pool
    pool.close()

    print("All experiments completed.")


