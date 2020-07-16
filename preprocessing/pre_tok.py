import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--datapath', type=str)
parser.add_argument('--savepath', type=str)

import keras
from keras_tokenizer import Tokenizer
import numpy as np


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
    savepath = args.savepath

    with open(datapath, 'r') as f:
        final_data = json.load(f)

    data = []
    labels = []

    for review in final_data:
        data.append(review['text'])
        labels.append(review['label'])

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels,
                         test_size=1/3,
                         stratify=labels,
                         random_state=1)

    skf = StratifiedKFold(n_splits=3, random_state=1)

    i = 0
    for train_index, validation_index in skf.split(test_data, test_labels):

        tokenizer = Tokenizer(num_words=10000)

        k_train_data = []
        k_train_labels = []

        k_train_data.extend(train_data)
        k_train_labels.extend(train_labels)
        for index in validation_index:
            k_train_data.append(test_data[index])
            k_train_labels.append(test_labels[index])

        tokenizer.fit_on_texts(k_train_data)
        tokenizer_json = tokenizer.to_json()
        with open(savepath + str(i) + '.json', 'w') as f:
            f.write(json.dumps(tokenizer_json))

        i += 1

if __name__ == '__main__':
    main()
    print('Done')
