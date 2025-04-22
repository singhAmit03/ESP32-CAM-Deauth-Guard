#@title Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("dataset_decision_tree.csv")  # Make sure this is your updated, feature-rich dataset

# Encode categorical features like MAC addresses and protocol
encoder = LabelEncoder()
df["Source"] = encoder.fit_transform(df["Source"])
df["Destination"] = encoder.fit_transform(df["Destination"])
df["Protocol"] = encoder.fit_transform(df["Protocol"])
df["Info"] = encoder.fit_transform(df["Info"].astype(str))  # In case it's textual

# Drop Time if not numerical or relevant (or convert to a feature if you want)
df.drop("Time", axis=1, inplace=True)

# Define features and target
X = df.drop("Label", axis=1)  # 'Label' should be your column with 0 (normal) and 1 (suspicious)
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Normal", "Suspicious"], rounded=True)
plt.show()

#additional Evaluation metrics

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, matthews_corrcoef
import matplotlib.pyplot as plt

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred)
print("Matthews Correlation Coefficient:", mcc)

