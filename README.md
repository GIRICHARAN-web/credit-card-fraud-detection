# Credit Card Fraud Detection

A machine-learning project for identifying potentially fraudulent credit-card transactions using the public Kaggle Credit Card Fraud Detection dataset. The project explores class imbalance, compares classification models, evaluates precision–recall performance, and provides a Streamlit dashboard for exploring transactions and model outputs.

> **Important:** This is an educational project, not a production fraud-prevention system. Predictions are probabilistic and should not be used as the sole basis for financial decisions.

## Project goals

- Explore a highly imbalanced transaction dataset.
- Compare Logistic Regression, Random Forest, tuned Random Forest, and Extra Trees.
- Evaluate models using Average Precision and fraud-class precision, recall, and F1—not accuracy alone.
- Explore score thresholds and the trade-off between false positives and missed fraud.
- Visualize model performance and feature importance in a Streamlit dashboard.
- Demonstrate a transaction-level sensitivity check to help inspect how changing one input affects the model score.

## Dataset

Download the **Credit Card Fraud Detection** dataset from Kaggle:

<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Place the downloaded `creditcard.csv` at `data/creditcard.csv`.

The dataset contains 284,807 transactions and 492 fraud examples. The `Class` column is the target: `0` indicates a legitimate transaction and `1` indicates fraud. `V1`–`V28` are transformed features; `Time` and `Amount` are also included.

The dataset is highly imbalanced, so accuracy alone can be misleading. Average Precision and fraud-class precision/recall are more informative for this task.

## Models and reported results

The following are results recorded during development. They are useful as a project log, not a guarantee that a fresh run will reproduce the exact same values. Results depend on the split, code version, and threshold.

| Model | Validation Average Precision | Test Average Precision |
|---|---:|---:|
| Logistic Regression | ~0.6857 | ~0.7575 |
| Random Forest baseline | ~0.7970 | ~0.8735 |
| Tuned Random Forest | ~0.8070 | ~0.8720 |
| Extra Trees | ~0.8090 | ~0.8760 |

The Extra Trees test result recorded during development used a separate held-out test set. Avoid repeatedly selecting models or thresholds based on the test set; use validation data for those choices and reserve the test set for final evaluation.

At one recorded Extra Trees operating point, the confusion matrix was `[[56848, 15], [14, 85]]`, corresponding to fraud precision of about 0.85, recall of about 0.86, and F1 of about 0.85. These threshold-dependent values should be reported alongside the threshold used and the number of fraud examples in the test set.

## Project structure

```text
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv          # Download separately; do not commit
├── models/                     # Optional saved model files
├── src/
│   ├── explore.py
│   ├── analyze.py
│   ├── train.py
│   ├── tune.py
│   └── extra_trees.py
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

Your local files may differ slightly. Keep the README in sync with the actual files in your repository.

## Setup

Python 3.10 or newer is a reasonable starting point.

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the Kaggle link above and place it in `data/creditcard.csv`. Do not commit the dataset to your public repository.

### 5. Run the analysis scripts

From the repository root:

```bash
python src/explore.py
python src/analyze.py
python src/train.py
python src/tune.py
python src/extra_trees.py
```

Each script may have its own outputs and assumptions; check the script before running it if your local version differs.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

If the dashboard cannot find the dataset or a saved model, check the file paths configured in `app.py`. Some versions may train a model when no saved model is available, which can take time.

## Evaluation approach

- **Average Precision (AP):** summarizes precision–recall performance across score thresholds.
- **Precision:** among transactions flagged as fraud, the fraction that are actually fraud.
- **Recall:** among actual fraud transactions, the fraction the model catches.
- **F1 score:** the harmonic mean of precision and recall.
- **Confusion matrix:** shows true/false positives and true/false negatives at a chosen threshold.

A threshold is a decision setting, not a universal constant. Lower thresholds usually flag more transactions, potentially catching more fraud while also creating more false alarms. Choose thresholds on validation data based on the intended trade-off.

## Explainability and limitations

The dashboard includes global feature-importance visualization and a transaction-level sensitivity check. A sensitivity check changes one feature at a time and observes how the score changes; it is **not** a causal explanation and does not fully account for interactions between features. Global feature importance also does not prove that a feature causes fraud.

Other limitations:

- The dataset is historical and may not represent current transaction patterns.
- `V1`–`V28` are transformed features and may not have straightforward real-world meanings.
- Performance can change with data drift, different splits, and different thresholds.
- A model score is not proof that a transaction is fraudulent.
- The dashboard is for demonstration and learning, not production deployment.

## Reproducibility notes

- The development experiments used stratified train/validation/test splits with `random_state=42`.
- Scaling was used for Logistic Regression; tree-based models were trained without feature scaling.
- Keep the test set separate from model and threshold selection.
- Re-run experiments before publishing updated metrics, and record the code version, split, threshold, and metrics together.

## Tools

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib / Seaborn (for analysis plots, where used)
- Streamlit
- Plotly (for dashboard visualizations)

Use `requirements.txt` as the authoritative list of packages installed for your current code.

## Future improvements

- Add automated tests for preprocessing and prediction paths.
- Add a reproducible training pipeline and save model metadata.
- Add calibration analysis and threshold selection on validation data.
- Add SHAP-based local explanations if the dependency and runtime cost are appropriate.
- Evaluate on newer data and monitor changes in data distribution.

## Acknowledgements

Dataset: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
