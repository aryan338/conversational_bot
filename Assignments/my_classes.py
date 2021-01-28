import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Dense, Lambda, GRU, Bidirectional, Conv1D, Conv2D, TimeDistributed, Permute, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow_addons.seq2seq import BeamSearchDecoder
from ds_utils.layers import SeqWiseBatchNorm

class DataGenerator(): 
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
             n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          X[i,] = np.load('data/' + ID + '.npy')

          # Store class
          y[i] = self.labels[ID]

      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y

ALPHABET_LENGTH = 30
eos_index  = 30
max_length = 10000 #

class DSModel(): 
  def __init__(self, input_shape, alpha = 0.3, beta = 0.2):
        self.input_shape = input_shape
        
        # Tunable hyperparams for net_loss
        self.alpha = alpha
        self.beta = beta

  def build(self, Name = "DeepSpeech2", num_conv = 3, num_rnn = 7, beam_width = 50): # TODO: Consider contentions with default beam width
        
      self.model =  Sequential(name = Name)
      # self.model.add(Input(shape = self.input_shape))

      # Conv Layers
      self.model.add(Conv2D(filters = 16, kernel_size = (3, 3), strides = 3, padding='same', input_shape = self.input_shape,  name = f"Conv1"))
      for i in range(1, num_conv):
          self.model.add(Conv2D(filters = 16, kernel_size = (3, 3), strides = 3, padding='same',name = f"Conv{i+1}"))
        
      # self.add(Conv1D(32, 3))

      # Conv2RNN : To be uncommented as per input dims, upon integration
      # self.model.add(Permute((0, 1, 2, 3)))
      self.model.add(Reshape(self.input_shape[-1], 16))

      # RNN Layers
      for i in range(num_rnn):
          self.model.add(Bidirectional(GRU(units = 800, return_sequences=True), name = f"RNN{i+1}")),
          self.model.add(SeqWiseBatchNorm(name = f"BatchNorm{i+1}"))

      # Beam Search Layer : TODO : Not sure if this is how its supposed to work
      BeamSearchDecoder(
          cell = self.model.get_layer(name = f"BatchNorm{i+1}"),
          beam_width = beam_width,
          output_layer = Dense(800)
      )
          
      # Final Layer
      self.model.add(TimeDistributed(Dense(units = ALPHABET_LENGTH, activation=softmax), name = "OutputLayer"))
      try:
          return self.model
      except:
          print("Couldn't build the model")
          return

  def ctc_find_eos(y_true, y_pred):
      # From SO : TODO : var init, predlength objective
      #convert y_pred from one-hot to label indices
      y_pred_ind = K.argmax(y_pred, axis=-1)

      #to make sure y_pred has one end_of_sentence (to avoid errors)
      y_pred_end = K.concatenate([y_pred_ind[:,:-1], eos_index * K.ones_like(y_pred_ind[:,-1:])], axis = 1)

      #to make sure the first occurrence of the char is more important than subsequent ones
      occurrence_weights = K.arange(start = max_length, stop=0, dtype=K.floatx())

      is_eos_true = K.cast_to_floatx(K.equal(y_true, eos_index))
      is_eos_pred = K.cast_to_floatx(K.equal(y_pred_end, eos_index))

      #lengths
      true_lengths = 1 + K.argmax(occurrence_weights * is_eos_true, axis=1)
      pred_lengths = 1 + K.argmax(occurrence_weights * is_eos_pred, axis=1)

      #reshape
      true_lengths = K.reshape(true_lengths, (-1,1))
      pred_lengths = K.reshape(pred_lengths, (-1,1))

      return K.ctc_batch_cost(y_true, y_pred, pred_lengths, true_lengths) + self.beta(pred_lengths) # Maybe a temp fix

  def net_loss(y_true, y_pred):
      # Summation log loss with ctc, word_count, lang model
      # Q(y) = log(p ctc (y|x)) + α log(p lm (y)) + β word_count(y)
      Loss = K.log(ctc_find_eos(y_true, y_pred)) #+ self.alpha*K.log(LangMod(y_pred)) #+ self.beta*word_count(y_pred) : need obviated with temp fix
      return Loss

  def summary(self):
      self.model.summary()

  def compile(self):
      self.model.compile(loss  = net_loss, optimizer = 'adam', metrics = ['word_error_rate'])
