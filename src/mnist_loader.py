
"""
load MNIST dataset
"""
import _pickle
import gzip
import os
import numpy as np

def load_data(rel_path):
    f = gzip.open(os.path.realpath(rel_path), 'rb')
    training_set, validation_set, testing_set = _pickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_set, validation_set, testing_set)