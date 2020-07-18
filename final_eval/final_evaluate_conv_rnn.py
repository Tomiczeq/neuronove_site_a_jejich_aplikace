import os
import json
import numpy as np
from keras_tokenizer import Tokenizer
from keras_tokenizer import tokenizer_from_json
from gensim.models import KeyedVectors

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--datapath', type=str)
parser.add_argument('--modelpath', type=str)
parser.add_argument('--word_vectors_path', type=str)

import tensorflow.keras.backend as K

from keras_sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

import tensorflow
from tensorflow.keras.models import load_model


class TextToSequence():

    def __init__(self):
        self.embeddings = None
        self.num_words = None
        self.embeddings_path = "../word_vectors/cz.txt"
        self.word_dict_path = None
        self.word_dict = None
        self.maxlen = None

    def load_embeddings(self, path=None, limit=None):
        model = KeyedVectors.load_word2vec_format(path,
                                                  unicode_errors='ignore',
                                                  binary=False,
                                                  limit=limit)
        self.num_words=limit
        vectors = model.wv.vectors
        word_dict = dict()
        for word in model.wv.vocab:
            index = model.wv.vocab[word].index
            word_dict[word] = index
        self.word_dict = word_dict
        self.embeddings = vectors
        self.w2v = model

    def convert_to_sequence(self, text):
        text = text.split(' ')
        sequence = []
        for word in text:
            if word in self.word_dict:
                sequence.append(self.word_dict[word])
        return sequence

    def convert(self, data, maxlen=200):
        sequences = self.process(data, maxlen)
        return sequences

    def process(self, data, maxlen):
        print('hah')
        sequences = []
        for i in range(len(data)):
            if i%100000 == 0:
                print(str(i))
            sequences.append(self.convert_to_sequence(data[i]))
        sequences = np.array(sequences)
        sequences = pad_sequences(sequences,
                             maxlen=maxlen,
                             padding='post')
        return sequences


def prepare_data(data):
    data = data.replace('__label__', '')
    data = data.split('\n')
    new_data = []
    labels = []
    for review in data:
        if review:
            new_data.append(review[2:])
            labels.append(int(review[0]))
    labels = np.array(labels)
    labels = np.where(labels == 1, 0, 1)

    return new_data, labels


def main():
    print('v4')
    args = parser.parse_args()

    datapath = args.datapath
    modelpath = args.modelpath
    embpath = args.embpath
    word_vectors_path = args.word_vectors_path
    word_vectors_dim = args.word_vectors_dim

    with open(datapath, 'r') as f:
        final_data = json.load(f)

    data = []
    labels = []

    for review in final_data:
        data.append(review['text'])
        labels.append(review['label'])
    labels = np.array(labels)

    K.clear_session()

    texts_to_sequences = TextToSequence()
    texts_to_sequences.load_embeddings(path=word_vectors_path,
                                           limit=30000)

    data_emb = texts_to_sequences.convert(data, maxlen=250)

    model = load_model(modelpath)
    history = model.evaluate(data_emb, labels, batch_size=1000)
    print(history)

    K.clear_session()

if __name__ == '__main__':
    main()
    print('Done')
