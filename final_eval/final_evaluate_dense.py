import os
import json
import numpy as np
import pandas as pd
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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class BatchGeneratorTFIDF(tensorflow.keras.utils.Sequence):

    def __init__(self, data, labels, tokenizer , batch_size=100):
        self.tokenizer = tokenizer
        self.data = data
        self.labels = labels
        self.indexes = np.arange(len(data))

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

        return data


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


    model = load_model(modelpath)
    predicted_raw = []
    step = 10000
    for i in range(0,len(data), step):
        tfidf_data = tokenizer.texts_to_matrix(data[i:i+step], mode='tfidf')
        prediction = model.predict(tfidf_data)
        predicted_raw.extend(list(prediction.reshape((len(prediction),))))
    predicted = np.round(predicted_raw)
    correct = np.array(labels)

    report = classification_report(correct, predicted, digits=4)
    conf_matrix = pd.crosstab(predicted, correct, rownames=['predicted'], colnames=['correct'])

    print(report)
    print(conf_matrix)

    K.clear_session()

if __name__ == '__main__':
    main()
    print('Done')

