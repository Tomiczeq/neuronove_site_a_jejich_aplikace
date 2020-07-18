# Neuronové sítě a jejich aplikace

Tento repozitář obsahuje zdrojové kódy a naměřené průběhy trénování k bakalařské práci.

Ve složce `preprocessing` Jsou skripty pro předzpracování textu.<br> 
Ve složce `random_search` Jsou skripty pro náhodné vyhledávání hyper-parametrů<br>
Ve složce `final_train` Jsou skripty pro trénování na celé datové sadě<br>
Ve složce `final_eval` Jsou skripty pro testování natrénovaných sítí<br>

Ve složce `data` je datová sada stažená z `https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M`.

Ve složce `word_vectors` Jsou předtrénované slovní vektory stažené z `https://wikipedia2vec.github.io/wikipedia2vec/pretrained/`, na kterých je provedena úprava textu a jsou zmenšené na 30 000 nejběžnějších slov

Ve složce `results` se nacházejí ještě dvě další složky:<br>
`random_search_results` obsahuje průběhy z náhodného vyhledávání<br>
`final_train_results` obsahuje průběhy z trénování na všech trénovacích recenzích.


## Zprovoznění v Google Colaboratory

### Naklonování repozitáře a připravení souborů s recenzemi 

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

### Instalace potřebných verzí knihoven
```
!pip install -r requirements_colab.txt
```

### Příprava trénovacích recenzí pro náhodné vyhledáváné

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

### Příprava všech trénovacích recenzí

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

### Příprava testovacích recenzí

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



## Plne propojena sit

### Náhodné vyhledávání

```
!python3 random_search/random_search_dense.py \
         --datapath=data/search_data.json \
         --savedir=random_search/results \
         --name=dense --iterations=1 \
         --n_splits=2
```

### Trénování na celé datové sadě

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

### Testování

```
!python3 final_eval/final_evaluate_dense.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/dense1000n0991 \
         --tokenizerpath=data/tokenizer1.json
```







## Konvolucni sit


### Náhodné vyhledávání 

```
!python3 random_search/random_search_conv.py \
         --iterations=1 --datapath=data/search_data.json \
         --savedir=random_search/results \
         --name=conv --n_splits=2 \
         --n_words=30000 --maxlen=250 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

### Trénování na celé datové sadě

```
!python3 final_train/final_conv.py \
         --name=conv95 --datapath=data/train_data.json \
         --results_savedir=final_train/results \
         --models_savedir=final_train/saved_models \
         --lr=0.0015 --rho=0.95 \
         --batch_size=1000 --epochs=1 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

### Testování

```
!python3 final_eval/final_evaluate_conv_rnn.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/conv950 \
         --word_vectors_path=word_vectors/en_300_30k.txt
```


## Rekurentní sít

### Náhodné vyhledávání

```
!python3 random_search/random_search_rnn.py \
         --iterations=1 --datapath=data/search_data.json \
         --savedir=random_search/results --name=rnn \
         --n_splits=2 --n_words=30000 --maxlen=250 \
         --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

### Trénování na celé datové sadě

```
!python3 final_train/final_rnn.py --name=rnn75 \
         --datapath=data/train_data.json \
         --results_savedir=final_train/results \
         --models_savedir=final_train/saved_models \
         --lr=0.0019 --rho=0.75 --batch_size=1000 \
         --epochs=1 --word_vectors_path=word_vectors/en_300_30k.txt \
         --word_vectors_dim=300
```

### Testování

```
!python3 final_eval/final_evaluate_conv_rnn.py \
         --datapath=data/test_data.json \
         --modelpath=final_train/saved_models/rnn750 \
         --word_vectors_path=word_vectors/en_300_30k.txt
```
