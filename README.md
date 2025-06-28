# 💳 Credit Card Fraud Detection with ROC-AUC Evaluation

This project uses *Decision Tree Classifier* and *Support Vector Machine (SVM)* models to detect fraudulent credit card transactions. The performance of the models is evaluated using the *ROC-AUC score*, a robust metric for binary classification problems.

---

## 📦 Libraries Used

- pandas – For data handling
- matplotlib – For visualization
- scikit-learn – For model training and evaluation
- train_test_split – For splitting dataset
- corr() – To select top 6 correlated features

---

## 🔍 Feature Selection

Used DataFrame.corr() to identify the *top 6 features most correlated* with the target (Class) and trained models on those features.

---

## 🧠 Models Trained

1. *Decision Tree Classifier*
2. *Support Vector Machine (SVM)*

Both models used random_state for reproducible results.

---

## 📊 Evaluation Metric: ROC-AUC Score

The *ROC-AUC (Receiver Operating Characteristic - Area Under Curve)* score was used to evaluate model performance.

### Why ROC-AUC?

- Measures how well the model distinguishes between classes
- Independent of classification threshold
- Especially useful for *imbalanced datasets*

### Code Example:

```python
from sklearn.metrics import roc_auc_score

# For Decision Tree
dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])

# For SVM (with probability=True)
svm_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

print("Decision Tree ROC-AUC:", dt_auc)
print("SVM ROC-AUC:", svm_auc)
