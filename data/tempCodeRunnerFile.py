import numpy as np, json

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

with open("data/processed/meta.json") as f:
    meta = json.load(f)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Meta:", meta)
