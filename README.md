# QA Extractive Model – MLOps Pipeline

## Project Overview

This project implements an **extractive Question Answering (QA) system** trained on the **SQuAD v2 dataset**.
The model predicts the **start and end positions of the answer span inside a context paragraph**.

The project also demonstrates a **complete MLOps pipeline** including:

* model training with **PyTorch**
* experiment tracking with **MLflow**
* dataset versioning with **DVC**
* automated training and evaluation via **GitHub Actions**
* automatic **model promotion (Staging → Production)** using the **MLflow Model Registry**
* containerization with **Docker**

---

# Architecture

## System Architecture

```
                ┌────────────────────┐
                │      GitHub        │
                │  (source code)     │
                └─────────┬──────────┘
                          │
                          │ Pull Request
                          ▼
                ┌────────────────────┐
                │  GitHub Actions CI │
                │                    │
                │ - run tests        │
                │ - train model      │
                │ - evaluate metrics │
                └─────────┬──────────┘
                          │
                          │ logs
                          ▼
                 ┌───────────────────┐
                 │      MLflow       │
                 │                   │
                 │ - experiments     │
                 │ - metrics         │
                 │ - artifacts       │
                 │ - model registry  │
                 └─────────┬─────────┘
                           │
                           │ promotion
                           ▼
                  ┌──────────────────┐
                  │  Model Registry  │
                  │                  │
                  │  None → Staging  │
                  │  Staging → Prod  │
                  └─────────┬────────┘
                            │
                            ▼
                     ┌───────────────┐
                     │ Docker Image  │
                     │ (GHCR)        │
                     └───────────────┘
```

---

# Model Architecture

The model implemented in `qa/model.py` is a **BiLSTM extractive QA model**.

The architecture follows a classical pipeline:

```
Input tokens
      │
Embedding layer
      │
BiLSTM encoder
      │
Context representation
      │
Two prediction heads
   ├── start position
   └── end position


# CI/CD Pipeline

The project uses **GitHub Actions** with a **three-stage workflow**.

---

## 1. Continuous Integration

Triggered on **Pull Requests to the development branch**.

Steps:

1. install dependencies
2. run unit tests
3. verify preprocessing and model components

Purpose:

* ensure code quality
* prevent broken code entering the pipeline

---

## 2. Dev → Staging

Triggered when a PR is merged from **dev → staging**.

Pipeline:

```
checkout code
↓
pull dataset (DVC)
↓
train model
↓
evaluate metrics
↓
quality gate
↓
promote model to STAGING
```

Quality gate example:

```
best_val_f1 >= 0.10
```

If the threshold is not reached the pipeline fails.

If successful:

```
MLflow Model Registry
stage = "Staging"
```

---

## 3. Staging → Production

Triggered when merging **staging → main**.

Steps:

```
run safety tests
↓
retrieve latest STAGING model
↓
promote to PRODUCTION
↓
build docker image
↓
push to container registry
```

Docker images are pushed to:

```
GitHub Container Registry (GHCR)
```

---

# Model Promotion

Model promotion is handled automatically using **MLflow Model Registry**.

Model lifecycle:

```
None
  │
  ▼
Staging
  │
  ▼
Production
```

### Promotion Rules

### Dev → Staging

Requirements:

* training successful
* F1 score above threshold

Result:

```
Model version promoted to Staging
```

---

### Staging → Production

Requirements:

* successful CI tests
* validated staging model

Result:

```
Model version promoted to Production
```

---

# Dataset Versioning (DVC)

Datasets are tracked with **DVC**.

Advantages:

* dataset versioning
* reproducibility
* remote storage

Dataset files:

```
data/train-v2.0.json
data/dev-v2.0.json
```

Mini datasets may be used for CI to reduce runtime.

---

# Reproducibility Instructions

## 1. Clone Repository

```
git clone <repository_url>
cd <repository_folder>
```

---

## 2. Install Dependencies

```
pip install -r requirements.txt
pip install -e .
```

---

## 3. Pull Dataset

```
dvc pull
```

---

## 4. Run Training

Example:

```
python training/train.py \
  --train_json data/train-mini.json \
  --val_json data/dev-mini.json \
  --epochs 3 \
  --batch_size 32
```

---

## 5. Disable MLflow (optional)

```
python training/train.py \
  --train_json data/train-mini.json \
  --no_mlflow
```

---

# Project Structure

```
project
│
├── qa
│   ├── data_utils.py
│   └── model.py
│
├── training
│   ├── train.py
│   └── eval.py
│
├── tests
│
├── data
│
├── artifacts
│
├── .github/workflows
│
└── README.md
```

---

# Technologies Used

* PyTorch
* MLflow
* DVC
* GitHub Actions
* Docker
* Python
