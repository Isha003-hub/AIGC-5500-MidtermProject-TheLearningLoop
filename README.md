# TheLearningLoop-DL

# AIGC 5500 — Deep Learning Optimizers: Midterm Project

**Course:** AIGC 5500: Advanced Deep Learning  
**Instructor:** Hossein Pourmodheji  
**Institution:** Humber Polytechnic — Faculty of Applied Science and Technology (FAST)  
**Team:** The Learning Loop  

| Member | Role |
|---|---|
| Ruchi Shah | Team Lead |
| Isha | Member |
| Krutik Babariya | Member |
| Dilkhush Yadav | Member |

---

## Project Overview

This project implements a systematic comparison of three modern deep learning optimizers — **Adam**, **RMSprop**, and **AdamW** — trained on the **KMNIST (Kuzushiji-MNIST)** dataset using a fixed feedforward neural network. The goal is to analyze how each optimizer behaves under different hyperparameter settings and draw evidence-based conclusions about their relative strengths and trade-offs.

---

## Repository Structure

```
aigc5500-midterm-optimizers/
│
├── report/                  # Written PDF report
│   └── report.pdf
│
├── slides/                  # Presentation slides
│   └── slides.pptx
│
├── src/                     # All source code
│   ├── data_loader.py       # Dataset download and preprocessing
│   ├── model.py             # Fixed feedforward network architecture
│   ├── train.py             # Training loop with cross-validation
│   ├── optimizer_search.py  # Hyperparameter search for all three optimizers
│   ├── evaluate.py          # Evaluation and metrics recording
│   ├── plots.py             # All required visualizations
│   └── main.ipynb           # End-to-end notebook with all results
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
| Output | 10 neurons, Softmax |
| Loss Function | Cross-Entropy Loss |

> The architecture is fixed across all runs — all differences between optimizer results are attributable solely to the optimizer and its hyperparameters.

---

## Dataset

**KMNIST (Kuzushiji-MNIST)** — loaded via `torchvision.datasets.KMNIST`

- 60,000 training samples / 10,000 test samples
- 28×28 greyscale images, 10 classes (Japanese Hiragana characters)
- More challenging than MNIST due to overlapping class boundaries

---

## Environment Setup

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

> **Note:** If you have a GPU, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## How to Reproduce Results

All results in the report can be reproduced by running the notebook from top to bottom with no modifications.

### Option 1 — Jupyter Notebook (recommended)

```bash
cd src
jupyter notebook main.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`).

### Option 2 — Command Line

```bash
cd src
python train.py          # Trains all optimizer configurations
python evaluate.py       # Records metrics
python plots.py          # Generates all required plots
```

---

## Random Seed

All experiments use a fixed random seed for reproducibility:

```python
RANDOM_SEED = 42
```

This seed is set at the top of every script and at the start of the notebook. It controls PyTorch, NumPy, and Python's `random` module.

---

## Optimizers Investigated

| Optimizer | Default LR | Key Hyperparameter Investigated |
|---|---|---|
| Adam | 0.001 | Learning rate, β₁ |
| RMSprop | 0.01 | Learning rate, decay factor α |
| AdamW | 0.001 | Learning rate, weight_decay |

Each optimizer is evaluated across **at least 4 hyperparameter configurations**, using **5-fold cross-validation** on the training set. The best configuration is then used to train a final model on the full training set.

---

## Library Versions

| Library | Version |
|---|---|
| Python | 3.9.x |
| PyTorch | 2.x |
| torchvision | 0.x |
| NumPy | 1.x |
| scikit-learn | 1.x |
| matplotlib | 3.x |

> Update this table with your exact versions before final submission. Run `pip freeze > requirements.txt` to capture them.

---

## Key Results (Summary)

> Fill this section in once experiments are complete.

| Optimizer | Best Config | Val Accuracy (mean ± std) | Test Accuracy | Epochs to 80% |
|---|---|---|---|---|
| Adam | lr=?, β₁=? | — | — | — |
| RMSprop | lr=?, α=? | — | — | — |
| AdamW | lr=?, wd=? | — | — | — |

---

## References

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015.*
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019.*
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5 - RMSProp. *COURSERA: Neural Networks for Machine Learning.*
- Clanuwat, T. et al. (2018). Deep learning for classical Japanese literature. *NeurIPS 2018 Workshop.*

---

## Academic Integrity

All submitted work complies with Humber Polytechnic's academic integrity policies.
