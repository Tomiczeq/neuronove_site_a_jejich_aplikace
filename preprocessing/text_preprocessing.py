import os
import json
from keras.preprocessing.text import text_to_word_sequence
    
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--filepath', type=str)
parser.add_argument('--savepath', type=str)

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
    filepath = args.filepath
    savepath = args.savepath

    print('loading {}'.format(os.path.abspath(filepath)))
    with open(filepath, 'r') as f:
        data = json.load(f)

    total = len(data)
    for i in range(len(data)):
        text = data[i]['text']
        text = preprocess(text)
        data[i]['text'] = text

    print('saving to {}'.format(os.path.abspath(savepath)))
    with open(savepath, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    main()
    print('Done')
