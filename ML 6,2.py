print("SHALINI A 24BAD409")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
 
df = pd.read_csv(r"C:\Users\SHALINI A\Downloads\diabetes_bagging.csv")
 
print("First 5 Rows:")
print(df.head())
 
print("\nColumns in Dataset:")
print(df.columns)
 
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1,
    random_state=42
)
 
ada_model.fit(X_train, y_train)
 
y_pred_ada = ada_model.predict(X_test)
y_prob_ada = ada_model.predict_proba(X_test)[:, 1]
 
ada_acc = accuracy_score(y_test, y_pred_ada)
print("\nAdaBoost Accuracy:", ada_acc)
 
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
 
gb_model.fit(X_train, y_train)
 
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
 
gb_acc = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", gb_acc)
 
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
 
auc_ada = auc(fpr_ada, tpr_ada)
auc_gb = auc(fpr_gb, tpr_gb)
 
plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.2f})")
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.2f})")
plt.plot([0,1], [0,1])
 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
 
importances = gb_model.feature_importances_
features = X.columns
 
indices = np.argsort(importances)[::-1]
 
plt.figure()
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
 
print(f"AdaBoost Accuracy        : {ada_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
 
if gb_acc > ada_acc:
    print("Gradient Boosting performs better.")
else:
    print("AdaBoost performs better or equal.")
