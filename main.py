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



def cnn_model(number_classes):
    cnn_model = models.Sequential([
        # Feature Extraction
        # layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),  # (16,16)

        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),

        # TODO:

        # Parameter Count:  2,097,280
        layers.Flatten(),

        #  32,896
        # layers.GlobalAveragePooling2D(),

        # Dense Layer
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(number_classes, activation='softmax')
        # layers.Dense(number_of_classes, activation='softmax')
    ])

images = []









