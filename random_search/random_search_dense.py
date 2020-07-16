import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras_tokenizer import Tokenizer

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--iterations', type=int)
parser.add_argument('--datapath', type=str)
parser.add_argument('--savedir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--n_splits', type=int)

import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

from keras.optimizers import RMSprop
from keras.optimizers import Adam


class RandomSearch():

    def __init__(self, k_folds=None, iterations=1):
        self.iterations = iterations
        self.k_folds = k_folds
        self.index = 0

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

        params['num_hidden_layers'] = random.randrange(1,5,1)
        params['units'] = []
        maximum = 500

        for i in range(params['num_hidden_layers']):
            minimum = 40 - i*10
            units = random.randrange(minimum, maximum)
            maximum = units
            params['units'].append(units)

        params['activation'] = random.choice(['sigmoid', 'relu'])
        params['dropout'] = random.uniform(0, 0.5)
        params['rho'] = random.uniform(0.6,0.95)
        params['learning_rate'] = random.uniform(0.00001, 0.001)
        params['batch_size'] = random.randrange(10,510,10)

        # fixed
        params['max_epochs'] = 80
        params['monitor_value'] = 'val_binary_accuracy'
        params['patience'] = 10

        print('training network with this hyper-parameters:')
        for name, value in params.items():
            print('{}: {}'.format(name, value))
            
        return params

    def get_model(self, params):
        model = Sequential()
        for i in range(params['num_hidden_layers']):
            if i == 0:
                model.add(Dense(params['units'][i],
                                input_shape=(10000,),
                                activation=params['activation']))
            else:
                model.add(Dense(params['units'][i],
                                activation=params['activation']))
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
    n_splits = args.n_splits

    if not os.path.exists(savedir):
        print('directory {} does not exists'.format(os.path.abspath(savedir)))
        print('creating ...')
        os.makedirs(savedir)

    print('loading {}'.format(os.path.abspath(datapath)))
    with open(datapath, 'r') as f:
        search_data = json.load(f)

    data = []
    labels = []

    for review in search_data:
        data.append(review['text'])
        labels.append(review['label'])

    skf = StratifiedKFold(n_splits=n_splits, random_state=1)

    print('preparing for random search ...')
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

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(train_data)

        train_data = tokenizer.texts_to_matrix(train_data, mode='tfidf')
        test_data = tokenizer.texts_to_matrix(test_data, mode='tfidf')

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        k_folds.append([train_data, train_labels, test_data, test_labels])

    for result in RandomSearch(k_folds, iterations=iterations):
        save_path = get_save_path(savedir, name)
        print('saving results to {}'.format(os.path.abspath(save_path)))
        with open(save_path, 'w') as f:
            f.write(json.dumps(result))


if __name__ == '__main__':
    main()
    print('Done')
