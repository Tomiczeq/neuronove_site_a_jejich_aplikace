import json
import argparse
import os
parser = argparse.ArgumentParser(description='')
parser.add_argument('--filepath', type=str)
parser.add_argument('--savepath', type=str)

def form(data):
    data = data.split('\n')
    new_data = []

    i = 0
    total_len = len(data)
    for review in data:
        if review:
            dct = dict()
            text = review[4:]
            label = int(review[1])
            label = label - 1
            dct['text'] = text
            dct['label'] = label
            new_data.append(dct)

    return new_data


def main():
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath

    print('loading {} '.format(os.path.abspath(filepath)))
    with open(filepath, 'r') as f:
        data = f.read()

    print('processing')
    data = form(data)

    print('saving to {}'.format(os.path.abspath(savepath)))
    with open(savepath, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    main()
    print('Done')


