
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc

from itertools import cycle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


print("Loading processed dataset...")

df = pd.read_csv("processed_dataset.csv")

print("\nDataset Shape:")
print(df.shape)



X = df.drop(columns=[" Label"])
y = df[" Label"]

print("\nFeature Shape:", X.shape)
print("Label Shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed.")



print("\nPredicting Test Data...")

y_pred = model.predict(X_test)

print("Prediction Completed.")



train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_pred)

test_accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Random Forest Result")
print("==============================")

print(f"\nTraining Accuracy : {train_accuracy:.6f}")
print(f"Test Accuracy     : {test_accuracy:.6f}")


# ==========================================
# Precision, Recall and F1 Score
# ==========================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"Precision        : {precision:.6f}")
print(f"Recall           : {recall:.6f}")
print(f"Weighted F1 Score: {f1:.6f}")



print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "BENIGN",
        "DDoS",
        "FTP-Patator",
        "SSH-Patator"
    ]
))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "BENIGN",
        "DDoS",
        "FTP",
        "SSH"
    ],
    yticklabels=[
        "BENIGN",
        "DDoS",
        "FTP",
        "SSH"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig("rf_confusion_matrix.png")

plt.close()

print("\nConfusion Matrix Saved.")



feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")

print(feature_importance.head(20))



feature_importance.to_csv(
    "rf_feature_importance.csv",
    index=False
)



top10 = feature_importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig("rf_feature_importance.png")

plt.close()

print("\nFeature Importance Graph Saved.")



joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("\nRandom Forest Model Saved.")



result = pd.DataFrame({

    "Model": [
        "Random Forest"
    ],

    "Training Accuracy": [
        train_accuracy
    ],

    "Test Accuracy": [
        test_accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "Weighted F1 Score": [
        f1
    ]

})

result.to_csv(
    "random_forest_result.csv",
    index=False
)

print("\nResult Saved.")

print("\nRandom Forest Finished Successfully.")

# ==========================================
# ROC Curve (Multi-Class)
# ==========================================

print("\nGenerating ROC Curve...")

# Probability prediction
y_score = model.predict_proba(X_test)

# Convert labels to binary
y_test_bin = label_binarize(
    y_test,
    classes=[0,1,2,3]
)

# Number of classes
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

# Plot ROC Curves

plt.figure(figsize=(8,6))

colors = cycle([
    "blue",
    "red",
    "green",
    "orange"
])

class_names = [
    "BENIGN",
    "DDoS",
    "FTP",
    "SSH"
]

for i, color in zip(range(n_classes), colors):

    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})"
    )

plt.plot(
    [0,1],
    [0,1],
    'k--'
)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Random Forest ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig("rf_roc_curve.png")

plt.close()

print("\nROC Curve Saved.")

print("\nAUC Scores")

for i in range(n_classes):

    print(f"{class_names[i]} : {roc_auc[i]:.6f}")