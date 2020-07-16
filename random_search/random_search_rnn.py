import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--iterations', type=int)
parser.add_argument('--datapath', type=str)
parser.add_argument('--savedir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--word_vectors_path', type=str)
parser.add_argument('--word_vectors_dim', type=int)
parser.add_argument('--n_splits', type=int)
parser.add_argument('--n_words', type=int)
parser.add_argument('--maxlen', type=int)

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

class TextToSequence():

    def __init__(self):
        self.embeddings = None
        self.num_words = None
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
        sequences = []
        for i in range(len(data)):
            sequences.append(self.convert_to_sequence(data[i]))
        sequences = np.array(sequences)
        sequences = pad_sequences(sequences,
                             maxlen=maxlen,
                             padding='post')
        return sequences


class RandomSearch():

    def __init__(self,
                 k_folds=None,
                 iterations=1,
                 embeddings=None,
                 maxlen=None,
                 n_words=None,
                 word_vectors_dim=None):
        self.iterations = iterations
        self.k_folds = k_folds
        self.index = 0
        self.embeddings = embeddings
        self.maxlen = maxlen
        self.n_words = n_words
        self.word_vectors_dim = word_vectors_dim

    def __len__(self):
        return self.iterations

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.iterations < self.index:
            raise StopIteration

        result = self.search()

        return result

    def search(self):
        params = self.get_params()
        result = dict()
        result['params'] = params
        result['results'] = []

        stop = EarlyStopping(monitor=params['monitor_value'],
                             patience=params['patience'],
                             mode='auto')

        for train_data, train_labels, test_data, test_labels in self.k_folds:
            K.clear_session()

            model = self.get_model(params)
            history = model.fit(train_data, train_labels,
                                epochs=params['max_epochs'],
                                batch_size=params['batch_size'],
                                validation_data=(test_data, test_labels),
                                callbacks=[stop]
                                )

            # json cannot serialize numpy float32
            for metric, values in history.history.items():
                values = np.array(values)
                history.history[metric] = values.tolist()

            result['results'].append(history.history)
            K.clear_session()

        return result

    def get_params(self):
        params = dict()
        params['num_hidden_layers'] = random.randrange(1,3,1)
        params['rnn_units'] = []
        maximum = 200

        for i in range(params['num_hidden_layers']):
            minimum = 40 - i*10
            units = random.randrange(minimum, maximum)
            maximum = units
            params['rnn_units'].append(units)

        params['activation'] = random.choice(['relu', 'sigmoid'])
        params['rnn_type'] = random.choice(['LSTM', 'GRU'])
        params['bidirectional'] = random.choice([True, False])
        params['units'] = random.randrange(5,100)
        params['dropout'] = random.uniform(0, 0.5)
        params['rho'] = random.uniform(0.6,0.95)
        params['learning_rate'] = random.uniform(0.0001, 0.005)
        params['batch_size'] = random.randrange(10,510,10)

        ## fixed
        params['max_epochs'] = 100
        params['monitor_value'] = 'val_binary_accuracy'
        params['patience'] = 10

        print('training network with this hyper-parameters:')
        for name, value in params.items():
            print('{}: {}'.format(name, value))

        return params

    def get_model(self, params):
        model = Sequential()
        model.add(Embedding(self.n_words,
                            self.word_vectors_dim,
                            input_length=self.maxlen,
                            weights=[self.embeddings],
                            trainable=False
                            )
                  )
        model.add(Dropout(params['dropout']))

        for i in range(params['num_hidden_layers']):
            if params['rnn_type'] == 'LSTM':
                rnn = LSTM
            else:
                rnn = GRU

            if params['bidirectional']:
                model.add(Bidirectional(rnn(params['rnn_units'][i], return_sequences=True)))
            else:
                model.add(rnn(params['rnn_units'][i], return_sequences=True))

            model.add(Dropout(params['dropout']))

        model.add(GlobalAveragePooling1D())
        model.add(Dropout(params['dropout']))

        model.add(Dense(params['units'], activation=params['activation']))
        model.add(Dropout(params['dropout']))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=params['learning_rate'], 
                                        rho=params['rho']
                                        ),
                      metrics=['binary_accuracy']
                      )

        return model
            

def get_save_path(savedir, name): 
        filenames = os.listdir(savedir)
        for i in range(len(filenames)):
            filenames[i] = filenames[i].split('_')
        matched_nums = []
        for filename in filenames:
            if filename[0] == name:
                matched_nums.append(int(filename[1]))

        if matched_nums:
            index = max(matched_nums) + 1
        else:
            index = 1

        save_path = os.path.join(savedir, name + '_' + str(index))
        return save_path



def main():
    args = parser.parse_args()
    datapath = args.datapath
    savedir = args.savedir
    name = args.name
    iterations = args.iterations
    word_vectors_path = args.word_vectors_path
    word_vectors_dim = args.word_vectors_dim
    n_splits = args.n_splits
    n_words = args.n_words
    maxlen = args.maxlen

    if not os.path.exists(savedir):
        print('directory {} does not exists'.format(os.path.abspath(savedir)))
        print('creating ...')
        os.makedirs(savedir)

    with open(datapath, 'r') as f:
        search_data = json.load(f)

    data = []
    labels = []

    for review in search_data:
        data.append(review['text'])
        labels.append(review['label'])

    skf = StratifiedKFold(n_splits=5, random_state=1)

    k_folds = []
    for train_index, test_index in skf.split(data, labels):

        train_data = []
        train_labels = []
        for index in train_index:
            train_data.append(data[index])
            train_labels.append(labels[index])

        test_data = []
        test_labels = []
        for index in test_index:
            test_data.append(data[index])
            test_labels.append(labels[index])

        texts_to_sequences = TextToSequence()
        texts_to_sequences.load_embeddings(path=word_vectors_path,
                                           limit=n_words
                                           )
        train_data = texts_to_sequences.convert(train_data, maxlen=maxlen)
        test_data = texts_to_sequences.convert(test_data, maxlen=maxlen)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        k_folds.append([train_data, train_labels, test_data, test_labels])

    for result in RandomSearch(k_folds,
                               iterations=iterations,
                               embeddings=texts_to_sequences.embeddings,
                               maxlen=maxlen,
                               n_words=n_words,
                               word_vectors_dim=word_vectors_dim):

        save_path = get_save_path(savedir, name)
        print('saving results to {}'.format(os.path.abspath(save_path)))
        with open(save_path, 'w') as f:
            f.write(json.dumps(result))


if __name__ == '__main__':
    main()
    print('Done')
