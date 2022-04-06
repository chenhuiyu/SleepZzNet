# SleepZzNet

This repo is the implementation of "***SleepZzNet: Sleep Stage Classification Using Single-Channel EEG Based on CNN and Transformer***", International Journal of Psychophysiology. [[paper\]](https://www.sciencedirect.com/science/article/pii/S016787602100670X)

By Huiyu Chen, Zhigang Yin, Peng Zhang, Panfei Liu

## Model Structure

![fig3-3_sleepZznetModel](E:\论文\2022-02-学位论文\初稿\Img\fig3-3_sleepZznetModel.png)

### Single EEG Channel ResNet

![单通道特征提取模块](E:\论文\2022-02-学位论文\初稿\Img\单通道特征提取模块.png)

### EEG+EOG ResNet

![多通道特征提取模块](E:\论文\2022-02-学位论文\初稿\Img\多通道特征提取模块.png)

## Get Started

### Environment

- python >=3.7.0
- tensorflow >= 2.7.0 (or compatible version to your develop env)
- numpy
- scikit-learn
- mne

### Data Preparation

Download [Sleep-EDF](https://archive.physionet.org/physiobank/database/sleep-edfx/). You can download SC subjects of Sleep-EDF using the following commands.

```
python download_sleepedf.py
```

### Train Model

- Train and Evaluation Sleep-EDF 

```
$ python main.py run_train.py
```

## Citation

If you find this project useful, we would be grateful if you cite our work as follows:

```
@article{CHEN2021S168,
title = {SleepZzNet: Sleep Stage Classification Using Single-Channel EEG Based on CNN and Transformer},
journal = {International Journal of Psychophysiology},
volume = {168},
pages = {S168},
year = {2021},
note = {Proceedings of the 20th World Congress of Psychophysiology (IOP 2021) of the International Organization of Psychophysiology (IOP)},
issn = {0167-8760},
doi = {https://doi.org/10.1016/j.ijpsycho.2021.07.464},
url = {https://www.sciencedirect.com/science/article/pii/S016787602100670X},
author = {Huiyu Chen and Zhigang Yin and Peng Zhang and Panfei Liu}
}
```