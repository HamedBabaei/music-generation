""" Defining the RNN Model """
import tensorflow as tf 
import helper
import numpy as np
from tqdm import tqdm

class RNN:
    
    def __init__(self):
        pass

    def complile(self, vocab_size, embedding_dim, rnn_units, 
                 batch_size, learning_rate, num_training_iterations, 
                 seq_length, checkpoint_prefix ):
        self.history = []
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
    
    def _LSTM(self, rnn_units):
        return tf.keras.layers.LSTM(rnn_units, 
                                    return_sequences=True, 
                                    recurrent_initializer='glorot_uniform',
                                    recurrent_activation='sigmoid',
                                    stateful=True,)
    
    def _set_optimizer(self, learning_rate):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
     
    
    def _compute_loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
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
        
    
    

    def song_generator(self, start_string, char2idx, idx2char, generation_length=1000):
        
        input_eval = [char2idx[s] for s in start_string] 
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []

        # Here batch size == 1
        self.model.reset_states()

        for i in tqdm(range(generation_length)):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            # Pass the prediction along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id]) # TODO 
            
        return (start_string + ''.join(text_generated))
    
    def load_model(self, checkpoint_dir):
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        
    def model_summary(self):
        return self.model.summary()
    
    def get_model(self, x):
        return self.model(x)
    
    def get_history(self):
        return self.history
    
    