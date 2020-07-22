""" Defining the RNN Model """
import tensorflow as tf 
import helper
import numpy as np
from tqdm import tqdm

class RNN:
    
    def __init__(self):
        self.history = []
    
    def complile(self, vocab_size, embedding_dim, rnn_units, 
                 batch_size, learning_rate, num_training_iterations, 
                 seq_length, checkpoint_prefix ):
        self.build_model(vocab_size, embedding_dim, rnn_units, batch_size)
        self._set_optimizer(learning_rate)
        self.num_training_iterations = num_training_iterations
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.checkpoint_prefix = checkpoint_prefix
    
    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        self.model = tf.keras.Sequential([
                # Layer 1: Embedding layer to transform indices into dense vectors 
                #          of a fixed embedding size
                tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
                
                # Layer 2: LSTM with `rnn_units` number of units. 
                self._LSTM(rnn_units),
                
                # Layer 3: Dense(fully-connected) layer that transforms the LSTM output
                #          into the vocabulary size. 
                tf.keras.layers.Dense(vocab_size)
            ])
        
    def _set_optimizer(self, learning_rate):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    def _compute_loss(self, labels, logits):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss

    @tf.function
    def train_step(self, x, y): 
        with tf.GradientTape() as tape:        
            y_hat = self.model(x)
            loss = self._compute_loss(y, y_hat) 

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def train(self, vectorized_songs):
        for iter in tqdm(range(self.num_training_iterations)):
            x_batch, y_batch = helper.get_batch(vectorized_songs, 
                                                self.seq_length, 
                                                self.batch_size)
            loss = self.train_step(x_batch, y_batch)
            self.history.append(loss.numpy().mean())
            if iter % 100 == 0:     
                self.model.save_weights(self.checkpoint_prefix)
        self.model.save_weights(self.checkpoint_prefix)

    def model_summary(self):
        return self.model.summary()
    
    def get_model(self, x):
        return self.model(x)
    
    def _LSTM(self, rnn_units):
        return tf.keras.layers.LSTM(rnn_units, 
                                    return_sequences=True, 
                                    recurrent_initializer='glorot_uniform',
                                    recurrent_activation='sigmoid',
                                    stateful=True,)
    
