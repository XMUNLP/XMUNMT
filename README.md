# RNNsearch
An implementation of RNNsearch using TensorFlow, the implementation is based on
with [DL4MT](https://github.com/nyu-dl/dl4mt-tutorial).

## Note
This repository is currently under major revision. We will release a new 
version soon. Stay tuned!


## Tutorial
This tutorial describes how to train a model on WMT17's EN-DE data using this
repository.

### Download Data
The preprocessed data can be found at 
[here](http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/).

### Data Preprocessing
1. Byte Pair Encoding
  * The most common approach to achieve open vocabulary is to use a technique
  called BPE. The codes of BPE can be found at 
  [here](https://github.com/rsennrich/subword-nmt).
  * To encode the training corpora using BPE, you need to generate BPE 
  operations first. The following command will create a file named "bpe32k" 
  which contains 32k BPE operations, along with two dictionaries named 
  "vocab.en" and "vocab.de".
  ```
  python subword-nmt/learn_joint_bpe_and_vocab.py 
    --input corpus.tc.en corpus.tc.de -s 32000 -o bpe32k 
    --write-vocabulary vocab.en vocab.de
  ```
  * You need to encode the training corpora, validation data and test data
  using the generated BPE operations and dictionaries. 
  ```
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en 
    --vocabulary-threshold 50 < corpus.tc.en > corpus.bpe32k.en
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.de 
    --vocabulary-threshold 50 < corpus.tc.de > corpus.bpe32k.de
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en 
    --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.bpe32k.en
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.de 
    --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.bpe32k.de
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en 
    --vocabulary-threshold 50 < newstest2017.tc.en > newstest2017.bpe32k.en
  ```
  
2. Build vocabulary
  * To train an NMT, you need to build a vocabulary file first. To build a 
  shared source and target vocabulary, following the steps below.
  ```
  cat corpus.bpe32k.en corpus.bpe32k.de > corpus.bpe32k.all
  python scripts/build_vocab.py --special "</s>:UNK" corpus.bpe32k.all 
    vocab.shared32k.txt
  ```
  * Note that the symbol "\</s\>" and "UNK" are reserved control symbols.
3. Shuffle corpus
  * It is beneficial to shuffle training corpora before training.
  ```
  python scripts/shuffle.py --corpus corpus.bpe32k.en corpus.bpe32k.de 
    --seed 1234
  ```
  * The above command will create two new files named "corpus.bpe32k.en.shuf"
  and "corpus.bpe32k.de.shuf".

### Training
  * Finally, we can start the training stage. The recommended hyper-parameters 
  are described below.
  ```
  python main.py train --model nmt  
    --corpus corpus.bpe32k.en.shuf corpus.bpe32k.de.shuf 
    --vocab vocab.bpe32k.shared.txt vocab.bpe32k.shared.txt 
    --embedding 512 --hidden 1024 --attention 2048
    --alpha 5e-4 --norm 5.0 --batch 128 --maxepoch 3 --seed 1234 
    --freq 1000 --vfreq 5000 --sfreq 50 --sort 20 --keep-prob 0.8
    --validation newstest2016.bpe32k.en --references newstest2016.bpe32k.de
    --gpuid 0
  ```
  * Change the argument of "--gpuid" to select GPU. The above command will 
  create a file named "nmt.autosave.pkl" every 1000 steps. The validation data
  will be evaluated every 5000 steps and the best model will saved to 
  "nmt.best.pkl" automatically.

### Decoding
  * The decoding command is quite simple.
  ```
    python main.py translate --model nmt.best.pkl < input > translation
  ```

## Benchmark
  * This section will be updated soon.
