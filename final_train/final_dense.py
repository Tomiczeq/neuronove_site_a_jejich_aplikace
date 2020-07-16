import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras_tokenizer import Tokenizer
from keras_tokenizer import tokenizer_from_json

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--datapath', type=str)
parser.add_argument('--results_savedir', type=str)
parser.add_argument('--models_savedir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--rho', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)

import tensorflow
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import RMSprop


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

        # na konci epochy nahodne zamicha indexy, aby
        # nebyla data pro dalsi epochu stejne serazena
        if index == self.steps_per_epoch - 1:
            np.random.shuffle(self.indexes)

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


def create_model(params):

    inp = Input(shape=(10000,))

    dense1 = Dense(139, activation='sigmoid')(inp)
    drop1 = Dropout(0.33)(dense1)

    dense2 = Dense(110, activation='sigmoid')(drop1)
    drop2 = Dropout(0.33)(dense2)

    dense3 = Dense(97, activation='sigmoid')(drop2)
    drop3 = Dropout(0.33)(dense3)

    dense4 = Dense(1, activation='sigmoid')(drop3)

    model = Model(inputs=inp, outputs=dense4)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=params['lr'], 
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
    results_savedir = args.results_savedir
    models_savedir = args.models_savedir
    name = args.name

    if not os.path.exists(results_savedir):
        print('directory {} does not exists'.format(os.path.abspath(results_savedir)))
        print('creating ...')
        os.makedirs(results_savedir)

    if not os.path.exists(models_savedir):
        print('directory {} does not exists'.format(os.path.abspath(models_savedir)))
        print('creating ...')
        os.makedirs(models_savedir)

    lr = args.lr
    rho = args.rho
    batch_size = args.batch_size
    epochs = args.epochs

    params = dict()
    params['lr'] = lr
    params['rho'] = rho
    params['batch_size'] = batch_size
    params['epochs'] = epochs

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

    k_folds = []
    i = 0
    for train_index, validation_index in skf.split(test_data, test_labels):
        K.clear_session()

        with open('data/tokenizer' + str(i) + '.json', 'r') as f:
            tokenizer_json = json.load(f)

        tokenizer = tokenizer_from_json(tokenizer_json)

        
        k_train_data = []
        k_train_labels = []

        k_train_data.extend(train_data)
        k_train_labels.extend(train_labels)
        for index in train_index:
            k_train_data.append(train_data[index])
            k_train_labels.append(train_labels[index])

        validation_data = []
        validation_labels = []
        for index in validation_index:
            validation_data.append(test_data[index])
            validation_labels.append(test_labels[index])

        k_train_labels = np.array(k_train_labels)
        validation_labels = np.array(validation_labels)

        models_save_path = os.path.join(models_savedir, name + str(i))
        checkpoint = ModelCheckpoint(
                models_save_path,
                monitor='val_binary_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)

        gen = BatchGeneratorTFIDF(k_train_data, k_train_labels, tokenizer, batch_size=params['batch_size'])
        val_gen = BatchGeneratorTFIDF(validation_data, validation_labels, tokenizer, batch_size=params['batch_size'])

        model = create_model(params)
        history = model.fit_generator(generator=gen,
                                      validation_data=val_gen,
                                      use_multiprocessing=True,
                                      workers=8,
                                      epochs=params['epochs'],
                                      callbacks=[checkpoint],
                                      )

        for metric, values in history.history.items():
            values = np.array(values)
            history.history[metric] = values.tolist()

        results = dict()
        results['params'] = params
        results['history'] = history.history

        results_save_path = os.path.join(results_savedir, name + str(i))
        with open(results_save_path, 'w') as f:
            f.write(json.dumps(results))

        i += 1

        K.clear_session()

if __name__ == '__main__':
    main()
    print('Done')
