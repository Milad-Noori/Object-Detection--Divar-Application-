import pandas as pd
import os
import warnings

from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import layers, models

import arabic_reshaper
from bidi.algorithm import get_display



images = []
