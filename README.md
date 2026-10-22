# Credit Card Fraud Detection — Imbalanced Classification

Portfolio project using the ULB credit card transaction dataset (284,807 transactions, 492 fraud cases). Includes EDA, stratified train/test split, Logistic Regression, SMOTE within a training-only pipeline, class-weighted Random Forest, confusion matrices, PR/ROC curves, and model comparison.

## Dataset source
[Credit Fraud — Dealing with Imbalanced Datasets (Kaggle)](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input?select=creditcard.csv)

Download `creditcard.csv` and place it in the same directory as the notebook. The dataset is excluded from Git.

## Run
1. `pip install -r requirements.txt`
2. Open `Credit_Card_Fraud_Detection.ipynb` in VS Code or Jupyter.
3. Select your Python environment and choose **Run All**.
4. Inspect `outputs/` for plots, model comparison, and a saved model.

## Evaluation
Report fraud-class precision, recall, F1, average precision, and ROC-AUC after running the notebook. Results must not be invented. SMOTE is applied only to training data; the test data is left untouched.

## Limitations
Educational demonstration using historical anonymized data; not a deployed payment risk system. Thresholds should be tuned on validation data in a production workflow.
