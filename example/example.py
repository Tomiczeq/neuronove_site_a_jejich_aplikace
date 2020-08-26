from keras_tokenizer import tokenizer_from_json  
from keras.preprocessing.text import text_to_word_sequence

import tensorflow
from tensorflow.keras.models import load_model


import json
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='models/dense/dense1000n0990')
parser.add_argument('--tokenizer', type=str, default='models/tokenizers/tokenizer0.json')


filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
def preprocess(text):
    text = text.lower()
    text = text_to_word_sequence(text, filters=filters)
    filtered_text = []
    for word in text:
        if word.isdigit():
            continue
        if len(word) < 3:
            continue
        filtered_text.append(word)

    filtered_text = ' '.join(filtered_text)

    return filtered_text


def main():
    args = parser.parse_args()
    modelpath = args.model
    tokenizerpath = args.tokenizer

    with open(tokenizerpath, 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    model = load_model(modelpath)

    while True:
        caption = input('Zadejte nadpis recenze: ')
        text = input('Zadejte text recenze: ')

        raw = caption + ' ' + text
        preprocessed = preprocess(raw)
        tfidf = tokenizer.texts_to_matrix([preprocessed], mode='tfidf')
        print(preprocessed)
        print(tfidf)

        prediction = model.predict(tfidf)

        if prediction[0][0] < 0.5:
            print('Tato recenze je negativní')
            print('Předpovězené skóre: ', prediction[0][0])
        else:
            print('Tato recenze je pozitivní')
            print('Předpovězené skóre: ', prediction[0][0])


if __name__ == '__main__':
    main()
