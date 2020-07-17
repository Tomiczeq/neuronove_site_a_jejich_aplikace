import os
import json
import numpy as np
from keras_tokenizer import Tokenizer
from keras_tokenizer import tokenizer_from_json

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--datapath', type=str)
parser.add_argument('--modelpath', type=str)
parser.add_argument('--tokenizerpath', type=str)

import tensorflow.keras.backend as K

import tensorflow
from tensorflow.keras.models import load_model


class BatchGeneratorTFIDF(tensorflow.keras.utils.Sequence):

    def __init__(self, data, labels, tokenizer , batch_size=100):
        self.tokenizer = tokenizer
        self.data = data
        self.labels = labels
        self.indexes = np.arange(len(data))
        np.random.shuffle(self.indexes)

        self.batch_size = batch_size
        self.steps_per_epoch = len(data) / self.batch_size
        if int(self.steps_per_epoch != self.steps_per_epoch):
            self.steps_per_epoch + 1
        self.steps_per_epoch = int(self.steps_per_epoch)

    def __len__(self):
        # pocet batchu na 1 epochu
        return self.steps_per_epoch

    def __getitem__(self, index):
        indexes = self.indexes[self.batch_size*index : self.batch_size*(index + 1)]
        indexes = np.sort(indexes)

        data = []
        labels = []
        for ind in indexes:
            data.append(self.data[ind])
            labels.append(self.labels[ind])

        data = self.tokenizer.texts_to_matrix(data, mode='tfidf')
        labels = np.array(labels)

        return data, labels


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
    args = parser.parse_args()

    datapath = args.datapath
    modelpath = args.modelpath
    tokenizerpath = args.tokenizerpath

    with open(datapath, 'r') as f:
        final_data = json.load(f)

    data = []
    labels = []

    for review in final_data:
        data.append(review['text'])
        labels.append(review['label'])

    K.clear_session()

    with open(tokenizerpath, 'r') as f:
        tokenizer_json = json.load(f)

    tokenizer = tokenizer_from_json(tokenizer_json)

    test_gen = BatchGeneratorTFIDF(data, labels, tokenizer, batch_size=1000)

    model = load_model(modelpath)
    prediction = model.predict_generator(generator=test_gen)
    print('Prediction accuracy on reviews from file test.csv : {}'.format(prediction))
    K.clear_session()

if __name__ == '__main__':
    main()
    print('Done')
