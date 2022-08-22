# Pre-training-Enhanced Spatial-Temporal Graph Neural Network For Multivariate Time Series Forecasting

[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
[![LICENSE](https://img.shields.io/github/license/zezhishao/BasicTS.svg)](https://github.com/zezhishao/BasicTS/blob/master/LICENSE)

Code for our SIGKDD'22 paper: "[Pre-training-Enhanced Spatial-Temporal Graph Neural Network For Multivariate Time Series Forecasting](https://arxiv.org/abs/2206.09113)".

The code is developed with [EasyTorch](https://github.com/cnstark/easytorch), an easy-to-use and powerful open source neural network training framework.

<img src="figure/STEP.png" alt="TheTable" style="zoom:42%;" />

All the training logs of the pre-training stage and the forecasting stage can be found in `train_logs/`.

> Multivariate Time Series (MTS) forecasting plays a vital role in a wide range of applications. Recently, Spatial-Temporal Graph Neural Networks (STGNNs) have become increasingly popular MTS forecasting methods. STGNNs jointly model the spatial and temporal patterns of MTS through graph neural networks and sequential models, significantly improving the prediction accuracy. But limited by model complexity, most STGNNs only consider short-term historical MTS data, such as data over the past one hour. However, the patterns of time series and the dependencies between them (i.e., the temporal and spatial patterns) need to be analyzed based on long-term historical MTS data. To address this issue, we propose a novel framework, in which STGNN is Enhanced by a scalable time series Pre-training model (STEP). Specifically, we design a pre-training model to efficiently learn temporal patterns from very long-term history time series (e.g., the past two weeks) and generate segment-level representations. These representations provide contextual information for short-term time series input to STGNNs and facilitate modeling dependencies between time series. Experiments on three public real-world datasets demonstrate that our framework is capable of significantly enhancing downstream STGNNs, and our pre-training model aptly captures temporal patterns.

## 1. Table of Contents

```text
config          -->     Training configs and model configs for each dataset
dataloader      -->     MTS dataset
easytorch       -->     EasyTorch
model           -->     Model architecture
checkpoints     -->     Saving the checkpoints according to md5 of the configuration file
datasets        -->     Raw datasets and preprocessed data
train_logs      -->     Our train logs.
TSFormer_CKPT   -->     Our checkpoints.
```

## 2. Requirements

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

### 3.1 Download Data

Download data from link [Google Drive](https://drive.google.com/drive/folders/1F7fEdXpnEQ75sxQval52jN4r3ZKoufGV?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1IWVhsvKpxHEb2CFLCR7P3A?pwd=w1wJ) to the code root directory.

Then, unzip data by:

```bash
unzip TSFormer_CKPT.zip
mkdir datasets
unzip raw_data.zip -d datasets
unzip sensor_graph.zip -d datasets 
rm *.zip
```
`TSFormer_CKPT/` contains the pre-trained model for each dataset.

You can also find all the training logs of the pre-training stage and forecasting stage in `training_logs/`.

### 3.2 Preprocess Data

```bash
python datasets/raw_data/$DATASET_NAME/generate_data.py
```

Replace `$DATASET_NAME` with one of `METR-LA`, `PEMS-BAY`, `PEMS04`.

The processed data is placed in `datasets/$DATASET_NAME`.

## 4. Train STEP based on a Pre-trained TSFormer

```bash
python main.py --cfg='config/$DATASET/forecasting.py' --gpu='0, 1'
# python main.py --cfg='config/METR-LA/forecasting.py' --gpu='0, 1'
# python main.py --cfg='config/PEMS-BAY/forecasting.py' --gpu='0, 1'
# python main.py --cfg='config/PEMS04/forecasting.py' --gpu='0, 1'
```

Replace `$DATASET_NAME` with one of `METR-LA`, `PEMS-BAY`, `PEMS04` as shown in the code above.

Configuration file
`config/$DATASET_NAME/forecasting.py` describes the forecasting configurations.

We use 2 GPU for forecasting stage as default, edit `GPU_NUM` property in the configuration file and `--gpu` in the command line to run on your own hardware.

Note that different GPU numbers lead to different real batch sizes, affecting the learning rate setting and the forecasting accuracy.

Our training logs are shown in `train_logs/Backend_metr.log`, `train_logs/Backend_pems04.log`, and `train_logs/Backend_pemsbay.log`.

## 5. Train STEP from Scratch

### 5.1 Pre-training Stage

```bash
python main.py --cfg='config/$DATASET/pretraining.py' --gpu='0'
# python main.py --cfg='config/METR-LA/pretraining.py' --gpu='0'
# python main.py --cfg='config/PEMS-BAY/pretraining.py' --gpu='0, 1, 2, 3, 4, 5, 6, 7'
# python main.py --cfg='config/PEMS04/pretraining.py' --gpu='0, 1'
```

Replace `$DATASET_NAME` with one of `METR-LA`, `PEMS-BAY`, `PEMS04` as shown in the code above.

Configuration file `config/$DATASET_NAME/pretraining.py` describes the pre-training configurations.

Edit the `BATCH_SIZE` and `GPU_NUM` in the configuration file and `--gpu` in the command line to run on your own hardware.

### 5.2 Forecasting Stage

Move your pre-trained model checkpoints to `TSFormer_CKPT/`.
For example:

```bash
cp checkpoints/TSFormer_200/9b4b52e25a30aabd21dc1c9429063196/TSFormer_180.pt TSFormer_CKPT/TSFormer_PEMS-BAY.pt
```

```bash
cp checkpoints/TSFormer_200/fac3814778135a6d46063e3cab20257c/TSFormer_147.pt TSFormer_CKPT/TSFormer_PEMS04.pt
```

```bash
cp checkpoints/TSFormer_200/3de38a467aef981dd6f24127b6fb5f50/TSFormer_030.pt TSFormer_CKPT/TSFormer_METR-LA.pt
```

Then train the downstream STGNN (Graph WaveNet) like in section 4.

## 6. Performance and Visualization
<!-- <img src="figures/Table3.png" alt="Table3" style="zoom:60.22%;" /><img src="figures/Table4.png" alt="Table4" style="zoom:51%;" /> -->
<img src="figure/MainResults.png" alt="TheTable" style="zoom:49.4%;" />

<img src="figure/Inspecting.jpg" alt="Visualization" style="zoom:25%;" />

## 7. More Related Works

- [D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting. VLDB'22.](https://github.com/zezhishao/D2STGNN)

- [BasicTS: An Open Source Standard Time Series Forecasting Benchmark.](https://github.com/zezhishao/BasicTS)

## 8. Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@inproceedings{DBLP:conf/kdd/ShaoZWX22,
  author    = {Zezhi Shao and
               Zhao Zhang and
               Fei Wang and
               Yongjun Xu},
  title     = {Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate
               Time Series Forecasting},
  booktitle = {{KDD} '22: The 28th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Washington, DC, USA, August 14 - 18, 2022},
  pages     = {1567--1577},
  publisher = {{ACM}},
  year      = {2022}
}
```
