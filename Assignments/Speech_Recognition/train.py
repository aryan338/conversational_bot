# To be run as a notebook
# Pipeline for training the model

from model import *
import tensorflow as tf 
from my_classes import DataGenerator
import os
from ds_utils.data_loader import *

#Data Generation
data = get_data()
dataset = DataGenerator(data[:, 0], data[:, 1])
"""
Data Augmentation
"""



# Building the model
input_shape = (32, 32, 32)

def main():
    model = DSModel(input_shape)
    model.build()
    model.compile()

    model.summary()
    input()

    # Training with sortagrad
    hist1 = model.fit(dataset, epochs = 1)
    hist2 = model.fit(dataset, epochs = 10)

    m1 = model.getModel()

    m1.save("DSM")

if __name__ == "__main__":
    main()