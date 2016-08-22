# RNNsearch
An implementation of RNNsearch using theano, the implementation is the same with
[Groundhog](https://github.com/lisa-groundhog/GroundHog)

## Usage

### Data Preprocessing
1. Prepare training corpus
2. Building vocabulary
  * Build source vocabulary
  ```
   python preprocess.py -d vocab.zh.pkl -v 30000 -b bintext.zh.pkl -p zh.txt
  ```
  * Build target vocabulary
  ```
   python preprocess.py -d vocab.en.pkl -v 30000 -b bintext.en.pkl -p en.txt
  ```
3. Shuffle corpus
  * Convert plain text to HDF5
  ```
    python preprocess.py -d vocab.unlimited.zh.pkl -b bintext.zh.pkl -p zh.txt
    python preprocess.py -d vocab.unlimited.en.pkl -b bintext.en.pkl -p en.txt
    python invert-dict.py vocab.unlimited.zh.pkl ivocab.unlimited.zh.pkl
    python invert-dict.py vocab.unlimited.en.pkl ivocab.unlimited.en.pkl
    python convert-pkl2hdf5.py bintext.zh.pkl bintext.zh.h5
    python convert-pkl2hdf5.py bintext.en.pkl bintext.en.h5
  ```
  * Shuffle
  ```
    python shuffle-hdf5.py bintext.zh.h5 bintext.en.h5 zh.shuf.h5 en.shuf.h5
  ```
  * Convert HDF5 to plain
  ```
    python hdf5_to_plain.py zh.shuf.h5 ivocab.unlimited.zh.pkl zh.shuf.txt
    python hdf5_to_plain.py en.shuf.h5 ivocab.unlimited.en.pkl en.shuf.txt
  ```
4. Partial sort, assuming 20 batches with batch size 128
  * Merge source and target
  ```
    python merget_split.py -m all.shuf.txt zh.shuf.txt en.shuf.txt
  ```
  * Create a new directory, split and sort
  ```
    split -d -l 2560 -a 5 all.shuf.txt
    ls | xargs -i% -n1 sort -g -k1 % -o %
    cat * > all.shuf.sort.txt
  ```
  * Split source and target
  ```
    python merget_split.py -s all.shuf.sort.txt zh.processed.txt en.processed.txt
  ```

### Training
```
  python rnnsearch.py train --corpus zh.processed.txt en.processed.txt \
    --vocab zh.vocab.pkl en.vocab.pkl --model nmt --embdim 620 620 \
    --hidden 1000 1000 1000 --maxhid 500 --deephid 620 --maxpart 2 \
    --alpha 5e-4 --norm 1.0 --batch 128 --maxepoch 5 --seed 1234 \
    --freq 1000 --vfreq 1500 --sfreq 50 --validate nist02.src \
    --ref nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
  ```
### Decoding
```
  python rnnsearch.py translate --model nmt.best.pkl < nist03.src > nist03.txt
```
### Resume training
```
  python rnnsearch.py train --model nmt.autosave.pkl
```

## How to get deterministic results
1. Add ```optimizer_excluding = cudnn``` to .theanorc
2. Use AdvancedIncSubtensor1 op instead of AdvancedIncSubtensor1_dev20 op,
see [here](https://github.com/Theano/Theano/issues/3029)
