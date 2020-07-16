import json
import argparse
import os
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='')
parser.add_argument('--filepath', type=str)
parser.add_argument('--savepath', type=str)
parser.add_argument('--size', type=float)


def main():
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath
    size = args.size

    print('loading {}'.format(os.path.abspath(filepath)))
    with open(filepath, 'r') as f:
        data = json.load(f)

    
    print('preparing data')
    new_data = []
    labels = []
    for review in data:
        new_data.append(review['text'])
        labels.append(review['label'])

    print('spliting')
    total = len(data)
    split = (total - size)/total
    data, _, labels, _ = train_test_split(new_data, labels,
                                          stratify=labels,
                                          random_state=1,
                                          test_size=split)

    new_data = []
    print('prepare for saving')
    for text, label in zip(data, labels):
        dct = dict()
        dct['text'] = text
        dct['label'] = label
        new_data.append(dct)

    print('saving to {}'.format(os.path.abspath(savepath)))
    with open(savepath, 'w') as f:
        f.write(json.dumps(new_data))


if __name__ == '__main__':
    main()
    print('Done')

        



