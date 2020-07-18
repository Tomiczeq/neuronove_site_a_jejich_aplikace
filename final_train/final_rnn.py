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
parser.add_argument('--word_vectors_path', type=str)
parser.add_argument('--word_vectors_dim', type=int)

import tensorflow
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import RMSprop

from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


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


def create_model(params=None,
                 embeddings=None,
                 word_vectors_dim=None):

    inp = Input(shape=(250,))

    emb = Embedding(30000,
                    word_vectors_dim,
                    input_length=250,
                    weights=[embeddings],
                    trainable=False)(inp)
    drop1 = Dropout(0.48)(emb)

    gru = Bidirectional(GRU(100, return_sequences=True))(drop1)
    drop2 = Dropout(0.48)(gru)

    glob = GlobalAveragePooling1D()(drop2)
    drop3 = Dropout(0.48)(glob)

    dens1 = Dense(8, activation='relu')(drop3)
    drop4 = Dropout(0.48)(dens1)

    dense = Dense(1, activation='sigmoid')(drop4)

    model = Model(inputs=inp, outputs=dense)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=params['lr'], 
                                    rho=params['rho']
                                    ),
                  metrics=['binary_accuracy']
                  )

    return model
            

def main():
    print('v3')
    args = parser.parse_args()

    datapath = args.datapath
    results_savedir = args.results_savedir
    models_savedir = args.models_savedir
    name = args.name
    kfold = args.kfold
    word_vectors_path = args.word_vectors_path
    word_vectors_dim = args.word_vectors_dim

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

        texts_to_sequences = TextToSequence()
        texts_to_sequences.load_embeddings(path=word_vectors_path,
                                           limit=30000)

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

        print('len val', len(validation_data))
        print('len train', len(k_train_data))

        k_train_labels = np.array(k_train_labels)
        validation_labels = np.array(validation_labels)

        train_data_emb = texts_to_sequences.convert(k_train_data, maxlen=250)
        validation_data_emb = texts_to_sequences.convert(validation_data, maxlen=250)

        models_save_path = os.path.join(models_savedir, name + str(i))
        checkpoint = ModelCheckpoint(
                models_save_path,
                monitor='val_binary_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)

        model = create_model(params=params,
                             embeddings=texts_to_sequences.embeddings,
                             word_vectors_dim=word_vectors_dim)
        results = dict()
        results['params'] = params
        results['history'] = dict()

        for e in range(params['epochs']):
            history = model.fit(train_data_emb, k_train_labels,
                                validation_data=(validation_data_emb, validation_labels),
                                epochs=1,
                                batch_size=params['batch_size'],
                                callbacks=[checkpoint],
                                )

            for metric, value in history.history.items():
                results['history'].setdefault(metric, [])
                results['history'][metric].append(float(value[0]))



            results_save_path = os.path.join(results_savedir, name + str(i))
            with open(results_save_path, 'w') as f:
                f.write(json.dumps(results))

        i += 1

        K.clear_session()

if __name__ == '__main__':
    main()
    print('Done')
