# Credit Card Fraud Detection with Random Forest

An end-to-end notebook for identifying rare fraudulent transactions using a Random Forest classifier, stratified cross-validation, and hyperparameter search. The project documents data validation, duplicate handling, model selection, and evaluation on a reserved test split.

## Results at a glance

On **212,795 reserved test transactions**, the selected model detected **272 of 355 fraudulent transactions**, with **20 legitimate transactions incorrectly flagged** at a fixed probability threshold of **0.5**.

| Test metric | Saved result |
| --- | ---: |
| Fraud precision | 93.15% |
| Fraud recall | 76.62% |
| Fraud F1-score | 0.8408 |
| Average precision (AP) | 0.8266 |
| ROC-AUC | 0.9680 |
| Accuracy | 99.95% |

Values are rounded from the saved outputs in `Credit_Card_Fraud_Detection(3).ipynb`. These are historical dataset results, not production performance. Accuracy alone is insufficient because fraud is rare. AP is a threshold-independent precision–recall summary; it is distinct from precision at threshold 0.5 and from trapezoidal PR-AUC.

## Dataset and validation

Dataset source: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- Input: `creditcard.csv`, with 284,807 rows and 31 columns.
- Features: `Time`, `Amount`, and `V1`–`V28`; target: `Class` (0 = legitimate, 1 = fraud).
- Validation checks: expected columns, numeric types, missing values, finite values, and binary labels.
- Removed 1,081 exact duplicate rows: 1,062 legitimate and 19 fraudulent records.
- Retained 283,726 transactions: 283,253 legitimate and 473 fraudulent, approximately 0.1667% fraud.
- Checked for identical feature vectors with conflicting labels and for exact feature overlap across the training/test split.

Without a reliable unique transaction ID, identical rows cannot be proven to be erroneous copies. This experiment keeps one copy of each exact row to prevent identical observations appearing across splits. Deduplication changes the evaluation population.

## Methodology

1. Validate the dataset and audit duplicates before splitting.
2. Create a stratified **25% training / 75% test split**, using `random_state=42`.
3. Perform exploratory analysis on training data only.
4. Tune a `RandomForestClassifier` using `RandomizedSearchCV` and shuffled **Stratified 5-Fold cross-validation** on training data.
5. Rank configurations by mean validation average precision and refit the winner on all training rows.
6. Evaluate the selected model on the reserved test split using a fixed 0.5 threshold.
7. Export metrics, plots, split indices, experiment metadata, and the fitted model.

| Split | Transactions | Legitimate | Fraud |
| --- | ---: | ---: | ---: |
| Training | 70,931 | 70,813 | 118 |
| Test | 212,795 | 212,440 | 355 |

This run uses a reduced training share to limit computation. It does not use SMOTE, scaling, or class weighting.

### Why use a 25% training / 75% test split?

The dataset was large enough that repeated Random Forest training took substantial time on the available hardware. I chose a smaller training share to make hyperparameter tuning manageable: 16 parameter combinations across five folds required **80 training fits**, followed by a final refit on all 70,931 training transactions. Reducing the training share reduced the data processed during these repeated fits. No timed benchmark was recorded, so no specific speedup is claimed.

Stratification approximately preserved the rare fraud proportion in both subsets. The remaining 212,795 transactions were reserved for evaluation; they were not used in hyperparameter search.

This split was a **computational trade-off**, not a demonstrated optimal split. Only 118 fraud examples were available for training, with approximately 94–95 fraud examples in each CV training fold. This limits learning and may contribute to variation across validation scores. A larger test set does not compensate for fewer training examples. With more computing resources, a larger training share and a fresh evaluation would be useful; changing the split would require reporting new results.

## Hyperparameter search

| Parameter | Values explored | Selected value |
| --- | --- | --- |
| `n_estimators` | 50, 100 | 100 |
| `max_depth` | 10, 20 | 10 |
| `min_samples_split` | 2, 5 | 2 |
| `min_samples_leaf` | 1, 2 | 1 |
| `max_features` | `sqrt` | `sqrt` |

Although `N_ITER=20` was requested, the search space contains only **16 unique combinations**. Scikit-learn evaluated all 16, resulting in **80 cross-validation fits plus one final refit**. The warning about this smaller search space does not indicate a failed run.

Winning mean CV AP: **0.7972 ± 0.1007** (standard deviation across five folds, not a confidence interval).

| Fold | Validation AP |
| --- | ---: |
| 1 | 0.698613 |
| 2 | 0.742275 |
| 3 | 0.926411 |
| 4 | 0.911648 |
| 5 | 0.706965 |

The winning CV score was used for model selection and is separate from the test AP of 0.8266.

## Confusion matrix and practical interpretation

| Actual class | Predicted legitimate | Predicted fraud |
| --- | ---: | ---: |
| Legitimate | 212,420 | 20 |
| Fraud | 83 | 272 |

- Of 292 transactions flagged, 272 were fraudulent: 93.15% precision.
- Of 355 actual fraud cases, 272 were detected and 83 were missed: 76.62% recall.
- These counts illustrate the trade-off between fraud detection and false alerts. No monetary savings, revenue increase, or production impact was measured.

## Financial impact simulation

A post-evaluation simulation estimates potential loss prevention using the test transactions' `Amount` values and explicit cost assumptions. The following results were produced by the additional financial simulation cell and reported after the model evaluation. **They are simulated estimates, not measured real-world savings.** All amounts below are in the same currency units as the dataset; no currency conversion is applied.

### Assumptions

- Baseline: without fraud detection, the full amount of every fraudulent test transaction is lost.
- Prevention rate: **80%** of the amount of correctly detected fraud can be prevented.
- Review cost: **5 currency units per alert**, including both true and false alerts.
- Additional false-alert cost: **0**, meaning customer inconvenience and other false-alert costs are excluded.
- Deployment, infrastructure, ongoing monitoring, and other operational costs are excluded.

### Simulation results

| Metric | Amount (dataset currency units) |
| --- | ---: |
| Total fraud transaction amount / baseline loss | 47,410.10 |
| Correctly detected fraud amount | 30,409.84 |
| Missed fraud amount | 17,000.26 |
| Potential prevented loss | 24,327.87 |
| Alert review cost | 1,460.00 |
| Additional false-alert cost | 0.00 |
| Estimated cost with model | 24,542.23 |
| **Estimated net savings versus no detection** | **22,867.87** |

### Calculation

```text
Alerts = 272 correctly detected fraud cases + 20 false alerts = 292
Potential prevented loss = 30,409.84 × 80% = 24,327.87
Review cost = 292 × 5 = 1,460.00
Estimated cost with model = 47,410.10 − 24,327.87 + 1,460.00
                          = 24,542.23
Estimated net savings = 24,327.87 − 1,460.00 = 22,867.87
```

Calculations use unrounded values internally; displayed amounts are rounded to two decimal places. Estimated cost with the model includes missed fraud, the unprevented share of correctly detected fraud, and review costs.

The model identified approximately **64.14% of the total fraudulent transaction amount**, compared with **76.62% of fraud cases**. These measure different things: transaction amounts vary. Under the stated assumptions, simulated net savings represent approximately **48.23% of the no-detection baseline loss**.

The prevention rate and review cost are illustrative inputs, not values learned or validated by the model. Different inputs change the savings estimate. This simulation compares against no detection, not an existing bank fraud system, and should not be presented as measured business impact or an annual savings forecast.

## Run locally

1. Download `creditcard.csv` from the dataset source.
2. Place the CSV and notebook in the same folder.
3. Open that folder in VS Code and select your Python notebook kernel.
4. Install dependencies in a notebook cell:

```python
%pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

5. Restart the kernel if needed, then run the notebook from top to bottom.

Run the additional financial simulation cell after model evaluation. It uses `X_test`, `y_test`, `test_predictions`, `OUTPUT_DIR`, pandas, and NumPy. Retraining is unnecessary if those variables remain available in the active kernel. The original uploaded notebook did not contain this additional cell; the financial results above come from the subsequently reported simulation output.

The notebook uses `Path.cwd()` to locate `creditcard.csv` and creates `outputs_cv/` automatically. Use the notebook folder as the working directory. In a terminal, use `python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib` instead of `%pip`.

`SEARCH_JOBS=2` limits concurrent search jobs, while each forest uses `n_jobs=1`. Set `SEARCH_JOBS=1` if memory is limited. Changing the split, parameters, or threshold creates a different experiment and requires fresh results.

## Generated outputs

| File in `outputs_cv/` | Contents |
| --- | --- |
| `duplicate_audit.csv` | Duplicate counts by class |
| `train_indices.csv`, `test_indices.csv` | Original row indices for each split |
| `randomized_search_cv_results.csv` | Scores and settings for candidate models |
| `final_test_metrics.csv` | Final test metrics |
| `simulated_financial_impact.csv` | Financial estimates generated by the additional simulation cell |
| `feature_importance.csv` | Impurity-based feature importances |
| `fraud_detection_model.joblib` | Fitted Random Forest |
| `experiment_summary.json` | Parameters, metrics, feature order, split sizes, and runtime versions |
| `01_class_distribution.png` | Training class distribution |
| `02_amount_distribution.png` | Training transaction amounts by class |
| `03_feature_correlations.png` | Training feature correlations with the target |
| `04_confusion_matrix.png` | Test confusion matrix |
| `05_test_curves.png` | Precision–recall and ROC curves |
| `06_feature_importance.png` | Top 15 feature importances |

These files are generated by running the notebook; this README does not bundle them. Keep the large dataset, virtual environment, and generated model out of GitHub. Only load trusted joblib files.

## Reproducibility notes and limitations

- The notebook's introductory text still mentions an 80/20 split; the executed code and saved outputs use **25/75**. This README follows the executed run.
- Some notebook descriptions mention 20 configurations and class-weight tuning; the actual run evaluated **16 configurations**, with no class-weight search. The metadata field `search_iterations` records the requested value 20, rather than the actual count 16.
- Only 118 fraud examples were available for training. The fold-to-fold AP variation should be considered when interpreting the score.
- The random split evaluates performance within this historical dataset, not future transactions under changing fraud patterns.
- If test rows influenced earlier development decisions, this is not a fully unseen external evaluation. A fresh temporal or external holdout would provide stronger evidence.
- The threshold was fixed at 0.5. Future threshold tuning should use training-only out-of-fold predictions or a separate validation split.
- Exact deduplication cannot exclude all near-duplicate or entity-related leakage. Feature importance describes model usage, not causation.
- This project is an offline notebook experiment; deployment, real-time monitoring, and measured financial savings are outside the implemented scope.

## Technology

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · joblib · Jupyter Notebook
