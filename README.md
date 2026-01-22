# CenterIR-EEG-Depression

This repository contains the official PyTorch implementation of the paper

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**

Last update : 2026/01/23

## Abstract

Emotion recognition from electroencephalography (EEG) offers promising opportunities for affective computing. However, conventional approaches often overlook the heterogeneity of auditory impairments. This study proposes a **frequency-aware deep learning framework** for EEG-based emotion recognition under simulated auditory conditions (Normal Hearing, Low-Frequency Loss Simulation, High-Frequency Loss Simulation).

The proposed model integrates:

1. **Multi-scale Convolutional Encoder:** Extracts localized time-frequency patterns with positional embeddings and cross-attention.


2. **Graph-Temporal Modeling:** Combines Graph Attention Networks (GAT) and Gated Recurrent Units (GRU) to model dynamic functional connectivity (PLV).


3. **Top-k Temporal Selection:** A classifier that aggregates outputs from the most emotionally salient segments.



Experiments achieved accuracies of **94.61% (HFsim)**, **90.00% (LFsim)**, and **78.08% (NH)**, demonstrating the effectiveness of frequency-aware modeling.


## File Structure

```bash
├── CenterIR.py         # Implementation of the proposed CenterIR loss function
├── run.py              # Main entry point to run
├── model.py            # CNN-Bi-LSTM architecture definition
├── train.py            # Training and validation procedures
├── requirements.txt    # Dependencies and version information
└── README.md           # Project documentation

```

## Dependencies

This project is implemented based on **PyTorch**.  
The following core dependencies are recommended to run the code properly.

> - python >= 3.10
> - torch = 2.7.0+cu118
> - scikit-learn = 1.6.1
> - numpy

All experimental dependencies and version details can be found in `requirements.txt`.


## Usage

### 1. Data Preparation

Both input data and targets are expected as NumPy (`.npy`) files.

- **Input**: NumPy array with shape `(N, 1, C, T)`
  - `N`: number of samples
  - `C`: number of channels
  - `T`: number of time points
- **Target**: NumPy array with shape `(N, 1)`

The dataset used in this study is not publicly available.
To verify that the training loop runs correctly, dummy data with the same format is provided.

Once the data is prepared, load the input and target data in `run.py` as follows:

```python
import numpy as np

input_np = np.load("path/to/input.npy")    # shape: (N, 1, C, T)
target_np = np.load("path/to/target.npy")   # shape: (N, 1)

```


### 2. Training

To train the model using 10-Fold Cross-Validation, run:

```python
python run.py

```

### 3. Hyperparameters

Key hyperparameters can be configured in the `hparams` dictionary within `main.py`:

```python
BATCH_SIZE = 32
Learning_Rate = 0.0001
EPOCHS = 100

## CenterIR hyperparameters
boundaries = 16
k = [1, 3, 5, 15]
CenterIR_lambda = 5e-8

```


## Citation

If you find this work useful in your research, please consider citing our paper:

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**

(The paper is currently under-review.)

---

*Note: This code is for research purposes only.*

  - 연구 간단소개
  - 아키텍처 cnn bi-lstm centerir 소개

  - 사용방법 : 넘파이 형태의 쉐입 뭐 이런 데이터를 준비하고요, 런 코드 돌립니다. 여기서 각각의 파라미터가 뭘 의미하냐면요 ~~~, 
