# Pareto-Optimal Fraud Rule Sets
Reproducing & extending Wen & Lou, "On Finding Bi-Objective Pareto-Optimal Fraud
Prevention Rule Sets for Fintech Applications" (KDD 2024).
Team: Sreya, Arthi, Yanin

## Setup
    python3 -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Check everything works (no download needed)
    python -m experiments.run --dataset synthetic --brute

## Data (NOT in git; put files in data/raw/)
| Dataset | File | How to get it |
|---|---|---|
| Default of Credit Card Clients | UCI_Credit_Card.csv | kaggle datasets download -d uciml/default-of-credit-card-clients-dataset --unzip |
| Credit Card Fraud | creditcard.csv | kaggle datasets download -d mlg-ulb/creditcardfraud --unzip |
| Give Me Some Credit | cs-training.csv | join competition on Kaggle, then: kaggle competitions download -c GiveMeSomeCredit |
| Bank Marketing | bank-full.csv | download from UCI (dataset 222), unzip |

## Run a real dataset
    python -m experiments.run --dataset default --brute
    python -m experiments.run --dataset creditcard
    python -m experiments.run --dataset credit
    python -m experiments.run --dataset bank

Outputs go to results/: <dataset>_front.png, <dataset>_summary.csv, <dataset>_example_rules.txt

## Who owns what
- Arthi: src/sca.py, src/spectral.py
- Sreya: src/pors.py
- Yanin: src/prep.py, src/baselines.py, src/eval.py
