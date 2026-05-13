# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import os
import numpy as np
import deeplake as hub
from sklearn.model_selection import train_test_split

os.makedirs("../dataset", exist_ok=True)

print("Loading Spoken MNIST dataset...")
ds = hub.load("hub://activeloop/spoken_mnist")

print("\nDataset Information:")
print(f"Total samples: {len(ds)}")
print(f"Tensors: {ds.tensors.keys()}")

spectrograms = []
labels = []

for sample in ds:
    spec = sample["spectrograms"].numpy()
    label = sample["labels"].numpy()

    spectrograms.append(spec)
    labels.append(label)

spectrograms = np.array(spectrograms)
labels = np.array(labels)


X_train, X_temp, y_train, y_temp = train_test_split(
    spectrograms, labels, test_size=0.30, random_state=42, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

np.save("../dataset/X_train.npy", X_train)
np.save("../dataset/y_train.npy", y_train)

np.save("../dataset/X_val.npy", X_val)
np.save("../dataset/y_val.npy", y_val)

np.save("../dataset/X_test.npy", X_test)
np.save("../dataset/y_test.npy", y_test)

print(f"Train: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")


