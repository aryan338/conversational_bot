# Pipeline for the deployment model, testing

import tensorflow as tf 
import tensorflow.keras as keras 
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Lambda, GRU, Bidirectional, Conv1D, Conv2D, TimeDistributed, Permute, Reshape
from ds_utils.layers import *
from ds_utils.data_loader import *

trained_model = keras.models.load_model("DSM")

test_model = Sequential(name = "DS_test")

def moving_avg_norm(input, moving_avg, weights):
    """
    params:
        moving avg - tuple of moving avg, variance
        weights : learned Lambda and Beta
    """
    mean, var = moving_avg
    L, b =  wieghts
    result = (input - mean)/tf.math.sqrt(var+1e-3)
    result = L*result + b
    return result


num_RNN = 7
num_Conv = 3

for i in range(num_Conv):
    c_layer = trained_model.get_layer(name = f"Conv{i+1}")
    test_model.add(c_layer)

for i in range(num_RNN):
    rnn_layer = trained_model.get_layer(name = f"RNN{i+1}")
    bn_layer = trained_model.get_layer(name = f"BatchNorm{i+1}")
    test_model.add(rnn_layer)
    bn_weights = bn_layer.get_weights()
    moving_avgerage = None # Load stored moving average from the layer (TODO)
    test_model.add(Lambda(lambda x: moving_avg_norm(x, bn_weights, moving_average)))


test_model.compile(loss  = DSModel.ctc_find_eos, optimizer = adam, metrics = ['word_error_rate'])

test_model.summary()

# Load test dataset
test_data = get_data("Librispeech/test-clean/") 

test_model.predict(test_data[:, 0])