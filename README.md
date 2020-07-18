# Neuronové sítě a jejich aplikace

Tento repozitář obsahuje zdrojové kódy naměřené průběhy trénování k bakalarské práci.

## Plne propojena sit

`!git clone https://github.com/Tomiczeq/neuronove_site_a_jejich_aplikace.git`

```
import os
os.chdir("/content/neuronove_site_a_jejich_aplikace")
```

```
!cat data/amazon_review_polarity_csv.tar.gz.part* > \
data/amazon_review_polarity_csv.tar.gz
```

```
!tar -zxvf data/amazon_review_polarity_csv.tar.gz && \
mv amazon_review_polarity_csv/* data/
```

```
!pip install -r requirements_colab.txt
```

```
!python3 preprocessing/format.py \
         --filepath=data/train.csv \
         --savepath=data/train.json
```

```
!python3 preprocessing/search_split.py \
         --filepath=data/train.json \
         --savepath=data/search_train.json \
         --size=50000
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/search_train.json \
         --savepath=data/search_data.json
```

```
!python3 random_search/random_search_dense.py \
         --datapath=data/search_data.json \
         --savedir=random_search/results \
         --name=dense --iterations=1 \
         --n_splits=2
```

```
!python3 preprocessing/format.py \
         --filepath=data/train.csv \
         --savepath=data/train.json
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/train.json \
         --savepath=data/train_data.json
```

```
!python3 preprocessing/pre_tok.py \
         --datapath=data/train.json \
         --savepath=data/tokenizer
```

```
!python3 final_train/final_dense.py \
         --name=dense1000n099 \
         --datapath=data/train_data.json \
         --results_savedir=final_train/results \
         --models_savedir=final_train/saved_models \
         --lr=0.001 --batch_size=1000 \
         --rho=0.99 --epochs=1
```

```
!python3 preprocessing/format.py \
         --filepath=data/test.csv \
         --savepath=data/test.json
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/test.json \
         --savepath=data/test_data.json
```

```
!python3 final_eval/final_evaluate_dense.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/dense1000n0991 \
         --tokenizerpath=data/tokenizer1.json
```

## Konvolucni sit

!git clone https://github.com/Tomiczeq/neuronove_site_a_jejich_aplikace.git

import os
os.chdir("/content/neuronove_site_a_jejich_aplikace")

!cat data/amazon_review_polarity_csv.tar.gz.part* > \
     data/amazon_review_polarity_csv.tar.gz

!tar -zxvf data/amazon_review_polarity_csv.tar.gz && \
     mv amazon_review_polarity_csv/* data/

!pip install -r requirements_colab.txt

!python3 preprocessing/format.py \
         --filepath=data/train.csv \
         --savepath=data/train.json

!python3 preprocessing/search_split.py \
         --filepath=data/train.json \
         --savepath=data/search_train.json \
         --size=50000

!python3 preprocessing/text_preprocessing.py \
         --filepath=data/search_train.json \
         --savepath=data/search_data.json

!python3 random_search/random_search_conv.py \
         --iterations=1 --datapath=data/search_data.json \
         --savedir=random_search/results \
         --name=conv --n_splits=2 \
         --n_words=30000 --maxlen=250 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300

!python3 preprocessing/text_preprocessing.py \
         --filepath=data/train.json \
         --savepath=data/train_data.json

!python3 final_train/final_conv.py \
         --name=conv95 --datapath=data/train_data.json \
         --results_savedir=final_train/results \
         --models_savedir=final_train/saved_models \\
         --lr=0.0015 --rho=0.95 \
         --batch_size=1000 --epochs=1 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300

!python3 preprocessing/format.py \
         --filepath=data/test.csv \
         --savepath=data/test.json

!python3 preprocessing/text_preprocessing.py \
         --filepath=data/test.json \
         --savepath=data/test_data.json

!python3 final_eval/final_evaluate_conv_rnn.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/conv950 \
         --word_vectors_path=word_vectors/en_300_30k.txt


## Rekurentni sit

```
!git clone https://github.com/Tomiczeq/neuronove_site_a_jejich_aplikace.git
```

```
import os
os.chdir("/content/neuronove_site_a_jejich_aplikace")
```

```
!cat data/amazon_review_polarity_csv.tar.gz.part* > \
     data/amazon_review_polarity_csv.tar.gz
```

```
!tar -zxvf data/amazon_review_polarity_csv.tar.gz && \
     mv amazon_review_polarity_csv/* data/
```

```
!pip install -r requirements_colab.txt
```

```
!python3 preprocessing/format.py \
         --filepath=data/train.csv \
         --savepath=data/train.json
```

```
!python3 preprocessing/search_split.py \
         --filepath=data/train.json \
         --savepath=data/search_train.json \
         --size=50000
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/search_train.json \
         --savepath=data/search_data.json
```

```
!python3 random_search/random_search_rnn.py \
         --iterations=1 --datapath=data/search_data.json \
         --savedir=random_search/results --name=rnn \
         --n_splits=2 --n_words=30000 --maxlen=250 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/train.json \
         --savepath=data/train_data.json
```

```
!python3 final_train/final_rnn.py --name=rnn75 \
         --datapath=data/train_data.json \
         --results_savedir=final_train/results \
         --models_savedir=final_train/saved_models \\
         --lr=0.0019 --rho=0.75 --batch_size=1000 \
         --epochs=1 --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

```
!python3 preprocessing/format.py \
         --filepath=data/test.csv \
         --savepath=data/test.json
```

```
!python3 preprocessing/text_preprocessing.py \
         --filepath=data/test.json \
         --savepath=data/test_data.json
```

```
!python3 final_eval/final_evaluate_conv_rnn.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/rnn750 \
         --word_vectors_path=word_vectors/en_300_30k.txt
```
