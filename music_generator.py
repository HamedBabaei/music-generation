"""
Music Generator
"""

import helper
from rnnmodel import RNN
import numpy as np


#loading training data
songs = helper.load_training_data(path='dataset/irish.abc')

# Join list of song strings
songs_joined = "\n\n".join(songs) 

# Find all unique characters in joined lists
vocab = sorted(set(songs_joined))
print("There are {} unique characters in the dataset with size of {}."
                               .format(len(vocab), len(songs_joined)))

# Mapping vocabs into char2idx and idx2char encoders/decoders
char2idx, idx2char = helper.get_mappings(vocab)

# Vectorize songs
vectorized_songs = helper.vectorize_string(songs_joined, char2idx)

# RNN model
rnn = RNN()
rnn.build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)

