
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow Version:", tf.__version__)
print("All libraries imported successfully.")




from google.colab import files

uploaded = files.upload()



import pandas as pd

df = pd.read_csv("processed_dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLabel Counts:")
print(df[" Label"].value_counts())


X = df.drop(columns=[" Label"])
y = df[" Label"]

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Scaling Completed")

print(X_train.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

# Input Layer
model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(32, activation="relu"))

# Output Layer
model.add(Dense(4, activation="softmax"))

model.summary()



model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("ANN Compiled Successfully")



from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=512,

    callbacks=[early_stop],

    verbose=1

)


import numpy as np

# Prediction probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

print("Prediction Completed")



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Training Accuracy
train_loss, train_accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

# Test Accuracy
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

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

print("\n==============================")
print("ANN Result")
print("==============================")

print(f"\nTraining Accuracy : {train_accuracy:.6f}")
print(f"Test Accuracy     : {test_accuracy:.6f}")
print(f"Precision         : {precision:.6f}")
print(f"Recall            : {recall:.6f}")
print(f"Weighted F1 Score : {f1:.6f}")

print("\nClassification Report\n")

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



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.title("ANN Confusion Matrix")

plt.tight_layout()

plt.show()



import pandas as pd

result = pd.DataFrame({

    "Model":[
        "ANN"
    ],

    "Training Accuracy":[
        train_accuracy
    ],

    "Test Accuracy":[
        test_accuracy
    ],

    "Precision":[
        precision
    ],

    "Recall":[
        recall
    ],

    "Weighted F1 Score":[
        f1
    ]

})

result.to_csv(
    "ann_result.csv",
    index=False
)

print("ANN Result Saved Successfully.")



print("\nGenerating ROC Curve...")

# Probability prediction
y_score = model.predict(X_test)

# Convert labels to binary
y_test_bin = label_binarize(
    y_test,
    classes=[0,1,2,3]
)

n_classes = y_test_bin.shape[1]

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

plt.title("ANN ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig("ann_roc_curve.png")

plt.close()

print("\nROC Curve Saved.")

print("\nAUC Scores")

for i in range(n_classes):

    print(f"{class_names[i]} : {roc_auc[i]:.6f}")