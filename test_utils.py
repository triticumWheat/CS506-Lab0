## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 0])
    vector2 = np.array([0, 1])
    
    result = cosine_similarity(vector1, vector2)
    
    assert result == 0, f"Expected 0, but got {result}"

def test_nearest_neighbor():
    point = np.array([1, 2])
    neighbors = np.array([[1, 3], [2, 1], [4, 5]])
    
    result = nearest_neighbor(point, neighbors)
    
    assert result == 0, f"Expected index 0, but got {result}"