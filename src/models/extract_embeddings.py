# src/models/extract_embeddings.py
import numpy as np
import os
import json
from sklearn.decomposition import PCA

PROCESSED = "data/processed"
OUT = "data/processed"

def load_processed():
    X = np.load(os.path.join(PROCESSED, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(PROCESSED, "y.npy"), allow_pickle=True)
    with open(os.path.join(PROCESSED, "labels.json")) as f:
        labels = json.load(f)
    return X, y, labels

def mean_pool_embedding(sequence):
    """
    sequence: (T, D) numpy array
    returns: (D,) embedding (mean over time)
    """
    return np.nanmean(sequence, axis=0)

def main(save_pca_dim=None):
    X, y, labels = load_processed()
    embeddings = []
    for seq in X:
        emb = mean_pool_embedding(seq)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)  # (N, D)

    if save_pca_dim:
        pca = PCA(n_components=save_pca_dim)
        embeddings = pca.fit_transform(embeddings)
        np.save(os.path.join(OUT, "pca_model.npy"), pca.components_)

    np.save(os.path.join(OUT, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUT, "emb_labels.npy"), y)
    print(f"[DONE] saved embeddings shape {embeddings.shape}")

if __name__ == "__main__":
    main(save_pca_dim=None)
