# RNNsearch
An implementation of RNNsearch using theano, the implementation is the same with
[Groundhog](https://github.com/lisa-groundhog/GroundHog)

## Usage

### Data Preprocessing
1. Build vocabulary
  * Build source vocabulary
  ```
  python scripts/buildvocab.py --corpus zh.txt --output vocab.zh.pkl
                               --limit 30000 --groundhog
  ```
  * Build target vocabulary
  ```
  python scripts/buildvocab.py --corpus en.txt --output vocab.en.pkl
                               --limit 30000 --groundhog
  ```
2. Shuffle corpus (Optional)
```
python scripts/shuffle.py --corpus zh.txt en.txt
```

### Build Dictionary (Optional)
If you want to use UNK replacement feature, you can build dictionary by
providing alignment file
```
python scripts/build_dictionary.py zh.txt en.txt align.txt dict.zh-en
```

### Training
```
  python rnnsearch.py train --corpus zh.txt.shuf en.txt.shuf \
    --vocab zh.vocab.pkl en.vocab.pkl --model nmt --embdim 620 620 \
    --hidden 1000 1000 1000 --maxhid 500 --deephid 620 --maxpart 2 \
    --alpha 5e-4 --norm 1.0 --batch 128 --maxepoch 5 --seed 1234 \
    --freq 1000 --vfreq 1500 --sfreq 50 --sort 20 --validate nist02.src \
    --ref nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
  ```
### Decoding
```
  python rnnsearch.py translate --model nmt.best.pkl < input > translation
```
### UNK replacement
```
  python rnnsearch.py replace --model nmt.best.pkl --text input translation \
    --dictionary dict.zh-en > newtranslation
```
### Resume training
```
  python rnnsearch.py train --model nmt.autosave.pkl
```
