import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#%matplotlib inline

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.saving import register_keras_serializable
#from keras.models import Sequential
#from keras.layers import Dense
#from keras import optimizers
#from keras import backend as K
#import keras.losses

import json
from sklearn.utils import shuffle
import os
import sys
import time

class ColabSimulator:
    def __init__(self):
        # Initialisierungscode, falls benötigt
        pass


    def cell_4(self):
        # Policy interface
        # !pip install tensorflow