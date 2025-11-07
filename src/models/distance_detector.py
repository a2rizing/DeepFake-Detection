# src/models/distance_detector.py
import numpy as np
import os
from scipy.spatial.distance import cosine, euclidean

PROCESSED = "data/processed"

def load_embeddings():
    emb = np.load(os.path.join(PROCESSED, "embeddings.npy"))
    labels = np.load(os.path.join(PROCESSED, "emb_labels.npy"))
    import json
    with open(os.path.join(PROCESSED, "labels.json")) as f:
        labels_map = json.load(f)
    return emb, labels, labels_map

def compare(i, j, metric="cosine"):
    emb, lab_ids, lab_map = load_embeddings()
    a = emb[i]
    b = emb[j]
    if metric == "cosine":
        score = 1 - cosine(a, b)  # similarity
    else:
        score = -euclidean(a, b)  # negative distance for similarity-like semantics
    return score

def rule_decision(similarity, threshold=0.75):
    # threshold tuneable; higher = stricter match required
    return "MATCH" if similarity >= threshold else "MISMATCH"

if __name__ == "__main__":
    emb, lab_ids, lab_map = load_embeddings()
    print("Loaded embeddings:", emb.shape)
    # compare sample 0 and 1 by default
    s = compare(0, 1, metric="cosine")
    print("Similarity (cosine):", s)
    print("Decision:", rule_decision(s, threshold=0.7))
