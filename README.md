# Stutter-Solver: End-to-end Cross-lingual Dysfluency Detection

## Architecture
<img src="resources/architecture.png" alt="image-20240321090057059" style="zoom: 20%; display: block; margin-right: auto; margin-left: 0;" />


## Datasets
We open sourced our three simulated datasets, **VCTK-Art**, **VCTK-Pro** and **AISHELL3-Pro**. Download link will be attached soon.

### Audio samples

check [Audio samples](https://drive.google.com/drive/folders/1LhYGUhAzkCT7P6JEYTqSV10Yq9ryjb__?usp=sharing)

## Pre-requisites
Please refer [environment.yml](environment.yml)

If you have Miniconda/Anaconda installed, you can directly use the command: `conda env create -f environment.yml`

## TTS-based Dysfluency Simulation

### English
We use [VITS](https://github.com/jaywalnut310/vits) as our TTS model.

0. Clone this repository

0. Download [VITS pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing), here we need `pretrained_vctk.pth` to achieve multi-speaker.
   1. create a folder `dysfluency_simulation/path/to`, and put the downloaded model into it.

0. Build Monotonoic Alignment Search
```sh
cd dysfluency_simulation/monotonic_align
python setup.py build_ext --inplace
```
3. Generate simulated speech
```sh
# Phoneme level
python generate_phn.py

# Word level
python generate_word.py
```

### Chinese Mandarin
Switch to [vits-chinese](https://github.com/eureka235/Stutter-Solver/tree/vits-chinese) branch.

We use [vits_chinese](https://github.com/PlayVoice/vits_chinese/tree/bert_vits_aishell3?tab=readme-ov-file) as our TTS model for Chinese simulation. Download checkpoints according to its `README` and place it in the specified path.

0. Build Monotonoic Alignment Search
```sh
cd monotonic_align
python setup.py build_ext --inplace
```
3. Generate simulated speech
```sh
python generate_dysfluency.py
```


