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


## Citation

If you find this work useful in your research, please consider citing our paper:

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**
(The paper is currently under review.)

---

*Note: This code is for research purposes only.*

  - 연구 간단소개
  - 아키텍처 cnn bi-lstm centerir 소개
  - 파일 구조 각 파일에 뭐있는지
  - 사용방법 : 넘파이 형태의 쉐입 뭐 이런 데이터를 준비하고요, 런 코드 돌립니다. 여기서 각각의 파라미터가 뭘 의미하냐면요 ~~~, 
  - 도움이 되엇다면 인용해주세용 (Manuscript under review)
