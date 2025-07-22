
#Stores all routes to relevant project directories

import os

# Automatically detect project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Standard subdirectories (parent directories)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

#Sub directories of standard directories
PRELIM_ANALYSIS_RESULTS_DIR = os.path.join(RESULTS_DIR, "preliminary_pair_analysis")
COMMUNITY_DETECTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "community_detection")
