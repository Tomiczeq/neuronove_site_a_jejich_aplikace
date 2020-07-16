import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--filepath', type=str)
parser.add_argument('--savepath', type=str)

import json
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

en_stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    filtered_text = []
    for word in lemmatized_sentence:
        if word not in en_stop_words:
            filtered_text.append(word)

    return " ".join(filtered_text)


def main():
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath

    print('loading {}'.format(os.path.abspath(filepath)))
    with open(filepath, 'r') as f:
        data = json.load(f)

    print('processing ...')
    total = len(data)
    for i in range(len(data)):
        text = data[i]['text']
        text.lower()
        lemmatized_text = lemmatize_sentence(text)
        data[i]['text'] = lemmatized_text

        print('{}/{}'.format(i, total), end='\r')

    print('saving to {}'.format(os.path.abspath(savepath)))
    with open(savepath, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    main()
    print('Done')

    







