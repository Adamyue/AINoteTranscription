# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:17:14 2020

@author: Austin Hsu
"""

# Fix OpenMP runtime conflict between PyTorch, NumPy, SciPy, and CuPy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

