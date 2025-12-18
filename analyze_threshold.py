import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======== LOAD MODEL AND FEATURES ========
MODEL_PATH = r"models/xgb2_solubility.joblib"
FEATURES_PATH = r"models/feature_cols.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

print("MODEL type:", type(model))
print("Model classes_:", model.classes_)
print("n_features_in_ (model expects):", model.n_features_in_)
print("len(feature_cols):", len(FEATURE_COLS))
print("feature_cols (first 10):", FEATURE_COLS[:10])

# ======== FEATURE BUILDER FUNCTION ========
def build_feature_vector(seq: str):
    seq = seq.strip().upper()
    aa_counts = {aa: seq.count(aa) / len(seq) for aa in FEATURE_COLS if aa.isalpha()}
    seq_length = len(seq)
    aromatic_fraction = sum(seq.count(x) for x in ['F', 'Y', 'W']) / seq_length

    # merge into a full feature vector in the same order as model expects
    features = []
    for col in FEATURE_COLS:
        if col == 'seq_length':
            features.append(seq_length)
        elif col == 'aromatic_fraction':
            features.append(aromatic_fraction)
        else:
            features.append(aa_counts.get(col, 0))
    return np.array(features).reshape(1, -1)

# ======== TEST SEQUENCES ========
test_sequences = {
    "GFP": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    "Hydrophobic Membrane": "MNGTEGPNFYVPFSNKTGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAFAVYMGVFSLAETNRFGAAHLP",
    "Short Cytosolic": "MKKIYFTKGQGPPAVPTTTGRSVPTIEVADKIVVGKPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"
}

# ======== EVALUATE AT MULTIPLE THRESHOLDS ========
thresholds = np.linspace(0.3, 0.9, 7)
y_true, y_pred = [], []
results = []

for name, seq in test_sequences.items():
    fv = build_feature_vector(seq)
    prob = model.predict_proba(fv)[0][1]
    print(f"\n{name} → Probability (soluble): {prob:.3f}")
    y_true.append(1 if "GFP" in name else 0)  # assume GFP is soluble, membrane is insoluble, cytosolic maybe soluble
    y_pred.append(prob)

for t in thresholds:
    pred_binary = [1 if p >= t else 0 for p in y_pred]
    acc = accuracy_score(y_true, pred_binary)
    prec = precision_score(y_true, pred_binary, zero_division=0)
    rec = recall_score(y_true, pred_binary, zero_division=0)
    f1 = f1_score(y_true, pred_binary, zero_division=0)
    cm = confusion_matrix(y_true, pred_binary)
    results.append((t, acc, prec, rec, f1))
    print(f"\nThreshold {t:.2f} — Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}")
    print("Confusion matrix:\n", cm)

# ======== RESULTS SUMMARY ========
df = pd.DataFrame(results, columns=["Threshold", "Accuracy", "Precision", "Recall", "F1"])
print("\n=== SUMMARY ===")
print(df)
print("\nSuggested threshold:", df.loc[df['F1'].idxmax(), 'Threshold'])
