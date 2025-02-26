# Before running the script, you need to install the numpy python third-party library in advance.
# Execute: pip install numpy

import numpy as np

def one_hot_encode(words, vocab):
    vector = np.zeros(len(vocab))
    for word in words:
        index = vocab.index(word)
        vector[index] = 1
    return vector


vocab = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
words = ["apple", "cherry", "kiwi", "mango", "apple"]

encoded_vector = one_hot_encode(words, vocab)
print(encoded_vector)