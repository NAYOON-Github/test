# CenterIR-EEG-Depression

This repository contains the official PyTorch implementation of the paper

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**

Last update : 2026/01/23

## 📝 Abstract

Emotion recognition from electroencephalography (EEG) offers promising opportunities for affective computing. However, conventional approaches often overlook the heterogeneity of auditory impairments. This study proposes a **frequency-aware deep learning framework** for EEG-based emotion recognition under simulated auditory conditions (Normal Hearing, Low-Frequency Loss Simulation, High-Frequency Loss Simulation).

The proposed model integrates:

1. **Multi-scale Convolutional Encoder:** Extracts localized time-frequency patterns with positional embeddings and cross-attention.


2. **Graph-Temporal Modeling:** Combines Graph Attention Networks (GAT) and Gated Recurrent Units (GRU) to model dynamic functional connectivity (PLV).


3. **Top-k Temporal Selection:** A classifier that aggregates outputs from the most emotionally salient segments.



Experiments achieved accuracies of **94.61% (HFsim)**, **90.00% (LFsim)**, and **78.08% (NH)**, demonstrating the effectiveness of frequency-aware modeling.

## 🏗️ Model Architecture

The framework consists of three sequential modules reflected in the code structure:

1. **Encoder Block (`model.py`)**:
* Stacked 2D CNN layers for hierarchical feature extraction.
* Positional embeddings to preserve temporal order.
* Cross-attention mechanism to integrate hierarchical features.


2. **Graph-Temporal Block (`model.py`)**:
* Constructs dynamic brain graphs using Phase Locking Value (PLV).
* **GAT:** Captures spatial dependencies among EEG channels.
* **GRU:** Models temporal evolution of graph embeddings.


3. **Classifier Block (`model.py` & `main.py`)**:
* **Top-k Selection:** Identifies and averages the *k* most informative time steps for final prediction.



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


## 🚀 Usage

### 1. Data Preparation

The code expects EEG features (`.npy`) and labels, along with PLV (Phase Locking Value) data.
Ensure your data is placed in the directory specified in `main.py` (default: `/home/coni/CONIRepo/...`).

You may need to modify the `data_dir` variable in `main.py`:

```python
# main.py
data_dir = "./data/"  # Update this path
features = torch.tensor(np.load(data_dir + 'HFsim_pre_features_v1.npy'), dtype=torch.float32)
labels = np.load(data_dir + 'HFsim_pre_labels_v1.npy')
plv_data = np.load(data_dir + 'HFsim/HFsim_PLV_all_fre_pooling.npy')

```

### 2. Training

To train the model using Stratified 10-Fold Cross-Validation, run:

```bash
python main.py

```

### 3. Hyperparameters

Key hyperparameters can be configured in the `hparams` dictionary within `main.py`:

* `seq_length`: 375 (Temporal length per sample)
* `feature_size`: 5 (Number of frequency bands: Delta, Theta, Alpha, Beta, Gamma)
* `in_channels`: 63 (Number of EEG electrodes)
* `gat_out_channels`: 32
* `gru_hidden_size`: 64
* `batch_size`: 50
* `learning_rate`: 1e-4

## 📊 Results

The proposed model was evaluated on a dataset of 48 participants under three auditory conditions.

| Condition | Accuracy (%) | F1-Score |
| --- | --- | --- |
| **HFsim** (High-Freq Loss Sim) | **94.61 ± 2.14** | **94.62** |
| **LFsim** (Low-Freq Loss Sim) | 90.00 ± 2.29 | 90.03 |
| **NH** (Normal Hearing) | 78.08 ± 1.15 | 78.18 |

*Table: Overall classification performance (Accuracy reported as Mean ± SD).*

```

## 👥 Authors & Contact

* **Seoyeon Kim** - [ksuyeon1102@catholic.ac.kr](mailto:ksuyeon1102@catholic.ac.kr)
* **Jihyun Lee** - [jihyunlee@hallym.ac.kr](mailto:jihyunlee@hallym.ac.kr)
* **Minji Lee** (Corresponding Author) - [minjilee@catholic.ac.kr](mailto:minjilee@catholic.ac.kr)

---

*Note: This code is for research purposes only.*

  - 연구 간단소개
  - 아키텍처 cnn bi-lstm centerir 소개
  - 파일 구조 각 파일에 뭐있는지
  - requirements 나는 이거는 파이썬, 토치, 사이킷런 중요한거 버전 공개하고 더 자세한 버전과 모듈들은 .txt에 있다고 해야지
  - 사용방법 : 넘파이 형태의 쉐입 뭐 이런 데이터를 준비하고요, 런 코드 돌립니다. 여기서 각각의 파라미터가 뭘 의미하냐면요 ~~~, 
  - 도움이 되엇다면 인용해주세용 (Manuscript under review)
