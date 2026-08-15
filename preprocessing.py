import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


print("Loading dataset...")

df = pd.read_csv("cicids2017_merged.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)


selected_labels = [
    "BENIGN",
    "DDoS",
    "FTP-Patator",
    "SSH-Patator"
]

df = df[df[" Label"].isin(selected_labels)]

print("\nFiltered Labels:")
print(df[" Label"].value_counts())

print("\nFiltered Dataset Shape:")
print(df.shape)


print("\nRemoving NaN and Infinity...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)


encoder = LabelEncoder()

df[" Label"] = encoder.fit_transform(df[" Label"])

print("\nEncoded Classes:")

for i, label in enumerate(encoder.classes_):
    print(f"{label} --> {i}")


df.to_csv("processed_dataset.csv", index=False)

print("\nProcessed dataset saved successfully!")

print("\nSaved File:")
print("processed_dataset.csv")

print("\nPreprocessing Completed Successfully.")