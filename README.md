# robust_backoff

### Requirements

#### Packages
- PyTorch==1.0.1
- AllenNLP==0.8.4
- simstring==1.1
- numpy==1.18.1

#### Datasets
- [GloVe.840B](https://nlp.stanford.edu/projects/glove/)
- [GloVe word frequency](https://github.com/losyer/compact_reconstruction)
- [CARD-660](https://pilehvar.github.io/card-660/)
- [TOEFL-Spell](https://github.com/EducationalTestingService/TOEFL-Spell)

### TOEFL dataset processing
```
$ cd toefl
$ python toefl.py
--wordemb_file glove.840B.300d.txt # word embeddings
--wordfreq_file freq_count.glove.840B.300d.txt.fix.txt # word frequency
--toefl_file TOEFL-Spell/Annotations.tsv # toefl dataset
--output_file output # output file
```

### Intrinsic evaluations
```
$ cd intrinsic
$ python pred.py
--cuda 0
--seed 0
--model log/model.pt
--wordemb_file glove.840B.300d.txt # word embeddings
--wordfreq_file freq_count.glove.840B.300d.txt.fix.txt # word frequency
--card_file card-660/dataset.tsv # card dataset
--toefl_file TOEFL-Spell/Annotations.tsv # toefl dataset
```

### Extrinsic evaluations

#### LSTM: NER and POS tagging
```
$ cd lstm/_sim
$ python ../run.py
--cuda 0
--seeds 0,1,2,3,4
```
The results will be stored in log folder.

#### BERT extension: NER and POS tagging
```
$ cd bert/_sim
$ python ../run.py
--cuda 0
--seeds 0,1,2,3,4
```
The results will be stored in log folder.

#### BERT extension: adversarial purtubations
```
$ cd robust
$ python ../run.py
--cudas 0
--seeds 0,1,2,3,4
```
The results will be stored in log folder.
