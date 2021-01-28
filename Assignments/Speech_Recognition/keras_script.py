import numpy as np

from keras.models import Sequential
from my_classes import *

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
#partition = # IDs
#labels = # Labels

# Generators
#training_generator = DataGenerator(partition['train'], labels, **params)
#validation_generator = DataGenerator(partition['validation'], labels, **params)
input_shape = 30
# Design model
model = DSModel(input_shape)
model.build()
model.compile()

model.summary()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
