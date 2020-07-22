import os
import re
from IPython.display import Audio
import numpy as np

def load_training_data(path):
    text = open(path, "r").read()
    return extract_song_snippet(text)

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs

def save_song_to_abc(song, filename="tmp"):
    open("{}.abc".format(filename), "w").write(song)
    return filename

def play_wav(wav_file):
    return Audio(wav_file)

def make_wav(song):
    abc = save_song_to_abc(song) + ".abc"
    res1 = os.system("abc2midi " + abc + " -o " + abc[:-4] + ".mid")
    res2 = os.system("timidity " + abc[:-4] + ".mid -Ow " + abc[:-4] + ".wav")
    res3 = os.system("rm " + abc + " " + abc[:-4] + ".mid")
    return (res1 and res2), abc[:-4]+".wav"

def play_song(song):
    res, wav = make_wav(song)
    if res == 0:
        return play_wav(wav)
    return "Not Successful"

def get_mappings(vocab):
    # Create a mapping from character to unique index.
    char2idx = {u:i for i, u in enumerate(vocab)}
    # Create a mapping from indices to characters. 
    idx2char = np.array(vocab)
    return char2idx, idx2char

def vectorize_string(string, char2idx):
    """ Vectorize the songs string """
    return np.array([char2idx[char] for char in string])

def get_batch(vectorized_songs, seq_length, batch_size):
    """ Batch definition to create training examples """
    
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1

    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    # construct a list of input sequences for the training batch
    input_batch = [vectorized_songs[i : i+seq_length] for i in idx]

    # construct a list of output sequences for the training batch
    output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

