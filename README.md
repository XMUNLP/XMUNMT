# XMUNMT
An open source Neural Machine Translation toolkit developed by the NLPLAB of Xiamen University.

## Features
* Multi-GPU support
* Builtin validation functionality


## Tutorial
This tutorial describes how to train an NMT model on WMT17's EN-DE data using this repository.

### Prerequisite
You must install TensorFlow (>=1.4.0) first to use this library. 

### Download Data
The preprocessed data can be found at 
[here](http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/).

### Data Preprocessing
1. Byte Pair Encoding
  * The most common approach to achieve open vocabulary is to use Byte Pair Encoding (BPE). The codes of BPE can be found at [here](https://github.com/rsennrich/subword-nmt).
  * To encode the training corpora using BPE, you need to generate BPE operations first. The following command will create a file named "bpe32k", which contains 32k BPE operations along with two dictionaries named "vocab.en" and "vocab.de".
  ```
  python subword-nmt/learn_joint_bpe_and_vocab.py --input corpus.tc.en corpus.tc.de -s 32000 -o bpe32k --write-vocabulary vocab.en vocab.de
  ```
  * You still need to encode the training corpora, validation set and test set using the generated BPE operations and dictionaries. 
  ```
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.bpe32k.en
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.bpe32k.de
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.bpe32k.en
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.bpe32k.de
  python subword-nmt/apply_bpe.py -c bpe32k --vocabulary vocab.en --vocabulary-threshold 50 < newstest2017.tc.en > newstest2017.bpe32k.en
  ```
  
2. Environment Variables
  * Before using XMUNMT, you need to add the path of XMUNMT to PYTHONPATH environment variable. Typically, this can be done by adding the following line to the .bashrc file in your home directory.
  ```
  PYTHONPATH=/PATH/TO/XMUNMT:$PYTHONPATH
  ```
  
2. Build vocabulary
  * To train an NMT, you need to build vocabularies first. To build a shared source and target vocabulary, you can use the following script:
  ```
  cat corpus.bpe32k.en corpus.bpe32k.de > corpus.bpe32k.all
  python XMUNMT/xmunmt/scripts/build_vocab.py corpus.bpe32k.all vocab.shared32k.txt
  ```
3. Shuffle corpus
  * It is beneficial to shuffle the training corpora before training.
  ```
  python XMUNMT/xmunmt/scripts/shuffle_corpus.py --corpus corpus.bpe32k.en corpus.bpe32k.de --seed 1234
  ```
  * The above command will create two new files named "corpus.bpe32k.en.shuf" and "corpus.bpe32k.de.shuf".

### Training
  * Finally, we can start the training stage. The recommended hyper-parameters are described below.
  ```
  python XMUNMT/xmunmt/bin/trainer.py
    --model rnnsearch
    --output train 
    --input corpus.bpe32k.en.shuf corpus.bpe32k.de.shuf
    --vocabulary vocab.shared32k.txt vocab.shared32k.txt
    --validation newstest2016.bpe32k.en
    --references newstest2016.bpe32k.de
    --parameters=device_list=[0],eval_steps=5000,train_steps=75000,
                 learning_rate_decay=piecewise_constant,
                 learning_rate_values=[5e-4,25e-5,125e-6],
                 learning_rate_boundaries=[25000,50000]
  ```
  * Change the argument of "device_list" to select GPU or use multiple GPUs. The above command will create a directory named "train".
    The best model can be found at "train/eval"

### Decoding
  * The decoding command is quite simple.
  ```
  python XMUNMT/xmunmt/bin/translator.py
    --models rnnsearch
    --checkpoints train/
    --input newstest2017.bpe32k.en
    --output test.txt
    --vocabulary vocab.shared32k.txt vocab.shared32k.txt
  ```
  
## Benchmark
|    Dataset    |    BLEU    |
| :------------ | :--------: |
|  WMT17 De-En  |    30.42   |

* More benchmarks will be added soon.


## Contact
This code is written by Zhixing Tan. If you have any problems, feel free to send an <a href="mailto:playinf@stu.xmu.edu.cn">email</a>.

## LICENSE
 BSD
