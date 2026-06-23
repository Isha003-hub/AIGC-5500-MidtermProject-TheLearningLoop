# TheLearningLoop-DL

# AIGC 5500 — Deep Learning Optimizers: Midterm Project

**Course:** AIGC 5500: Advanced Deep Learning  
**Instructor:** Hossein Pourmodheji  
**Institution:** Humber Polytechnic — Faculty of Applied Science and Technology (FAST)  
**Team:** The Learning Loop  

| Member | Role |
|---|---|
| Isha | Team Lead |
| Ruchi Shah | Member — Adam optimizer track |
| Dilkhush Yadav | Member — AdamW optimizer track |
| Krutik Babariya | Member — RMSprop optimizer track |

---

## Project Overview

This project implements a systematic comparison of three modern deep learning optimizers — **Adam**, **RMSprop**, and **AdamW** — trained on the **KMNIST (Kuzushiji-MNIST)** dataset using a fixed feedforward neural network. The goal is to analyze how each optimizer behaves under different hyperparameter settings and draw evidence-based conclusions about their relative strengths and trade-offs.

A structured hyperparameter search with **5-fold cross-validation** was used to identify the best configuration per optimizer. Final models were trained for 20 epochs and evaluated on a held-out test set.

---

## Repository Structure

```
TheLearningLoop-DL/
│
├── report/
│   └── AIGC5500_Midterm_Project_Report.docx   # Written report
│
├── slides/
│   └── AIGC5500_Midterm_Project_ppt.pptx      # Presentation slides
│
├── AIGC5500_Midterm_Master_Short_runned.ipynb  # End-to-end notebook with all results
│
└── README.md
```

---

## Model Architecture

| Layer | Configuration |
|---|---|
| Input | 784 neurons (flattened 28×28) |
| Hidden 1 | 128 neurons, ReLU |
| Hidden 2 | 64 neurons, ReLU |
| Output | 10 neurons (raw logits; softmax applied by loss) |
| Loss Function | Cross-Entropy Loss |

> The architecture is **fixed across all runs** — all differences between optimizer results are attributable solely to the optimizer and its hyperparameters.

---

## Dataset

**KMNIST (Kuzushiji-MNIST)** — loaded via `torchvision.datasets.KMNIST`

| Property | Value |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28×28 greyscale |
| Classes | 10 (cursive Japanese hiragana characters) |
| Difficulty | Harder than MNIST — class boundaries overlap |

---

## Environment

| Library | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| NumPy | 1.x |
| scikit-learn | 1.x |
| matplotlib | 3.x |

> Experiments were run on a **CUDA GPU** (cu128 build). To run on CPU only, install the CPU build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) — results may differ slightly due to floating-point non-determinism across devices.

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## How to Reproduce Results

All results in the report can be reproduced by running the notebook from top to bottom with no modifications.

```bash
jupyter notebook AIGC5500_Midterm_Master_Short_runned.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## Random Seed

All experiments use a fixed random seed for reproducibility:

```python
RANDOM_SEED = 42
```

This seed is set at the top of every script and at the start of the notebook. It controls PyTorch, NumPy, and Python's `random` module.

---

## Optimizers Investigated

| Optimizer | Default LR | Hyperparameters Searched |
|---|---|---|
| Adam | 0.001 | Learning rate ∈ {0.1, 0.01, 0.001, 0.0001}, β₁ ∈ {0.8, 0.9, 0.95} |
| AdamW | 0.001 | Learning rate ∈ {0.1, 0.01, 0.001, 0.0001}, weight_decay ∈ {0.001, 0.01, 0.1} |
| RMSprop | 0.01 | Learning rate ∈ {0.01, 0.005, 0.001, 0.0001}, α ∈ {0.9, 0.95, 0.99} |

Each optimizer was evaluated across **at least 4 hyperparameter configurations** using **5-fold cross-validation** on the training set. The best configuration (highest mean CV accuracy) was then used to train a final model on the full training set for **20 epochs**.

---

## Key Results

| Optimizer | Best Config | CV Acc (mean ± std) | Test Acc | Train Acc | Gen. Gap | Test Loss | Epoch to 80% |
|---|---|---|---|---|---|---|---|
| Adam | lr=0.001, β₁=0.8 | 95.22% ± 0.05 | 89.66% | 99.55% | 9.89 | 0.7584 | 1 |
| **AdamW ★** | **lr=0.001, wd=0.01** | **95.09% ± 0.13** | **89.94%** | **99.64%** | **9.70** | **0.656** | **1** |
| RMSprop | lr=0.001, α=0.99 | 95.14% ± 0.15 | 89.76% | 99.62% | 9.86 | 0.8221 | 1 |

**★ Winner: AdamW** — highest test accuracy (89.94%), smallest generalization gap (9.70), and lowest test loss (0.656).

> **Key finding:** Learning rate was the most impactful hyperparameter across all three optimizers. All three peaked at lr=0.001. AdamW with lr=0.1 collapsed to ~16.5% CV accuracy (near-random), demonstrating high sensitivity to a badly-chosen learning rate.

---

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015.*
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019.*
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5 — RMSProp. *Coursera: Neural Networks for Machine Learning.*
- Clanuwat, T. et al. (2018). Deep learning for classical Japanese literature. *NeurIPS 2018 Workshop.*

---

## Academic Integrity

All submitted work complies with Humber Polytechnic's academic integrity policies.
