import os
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser(description='nevim')
parser.add_argument('--filename', type=str)
parser.add_argument('--save_as', type=str)
parser.add_argument('--num_words', type=int)

filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
translate_map = str.maketrans(filters, ' ' * len(filters))


def main():
    args = parser.parse_args()
    filename = args.filename
    save_as = args.save_as
    num_words = args.num_words

    word_dict = OrderedDict()
    with open(os.path.join(filename), 'r', errors='ignore') as f:
        i = 0
        cp = 0
        total_words, dim = f.readline().split(' ')
        dim = int(dim)

        while i < num_words:
            line = f.readline()
            line = line.rstrip().split(' ')

            word = line[0]
            vector = line[1:]

            if 'ENTITY' in word:
                cp +=1
                continue
            if word.isdigit():
                continue
            if len(word) < 3:
                continue

            word = word.lower()
            word = word.translate(translate_map)
            word = word.strip()

            word_dict[word] = vector
            i += 1
            print(i, end='\r')
        print()
    
    with open(save_as, 'w') as f:
        f.write('{} {}'.format(num_words, dim))
        f.write('\n')
        for word, vector in word_dict.items():
            vector.insert(0, word)
            line = ' '.join(vector)
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    main()
    print('Done')


