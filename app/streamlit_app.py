# app/streamlit_app.py
import streamlit as st
from streamlit.components.v1 import html
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import os

# -----------------------------
# Config / Paths (relative for deploy)
# -----------------------------
MODEL_PATH = "models/xgb2_solubility.joblib"
FEATURE_COLS_PATH = "models/feature_cols.json"
ROC_PATH = "reports/roc.png"               # optional: create in notebook and these will show
CONF_PATH = "reports/confusion_matrix.png" # optional
st.set_page_config(page_title="Protein Solubility Predictor", layout="wide")

# -----------------------------
# Helper: session state for sample buttons
# -----------------------------
if "seq_input" not in st.session_state:
    st.session_state["seq_input"] = ""

# -----------------------------
# Load model + feature columns (cached)
# -----------------------------
@st.cache_resource
def load_model_and_cols():
    if not Path(MODEL_PATH).exists() or not Path(FEATURE_COLS_PATH).exists():
        raise FileNotFoundError(f"Missing model or feature_cols. Expected:\n{MODEL_PATH}\n{FEATURE_COLS_PATH}")
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        cols = json.load(f)
    return model, cols

try:
    model, FEATURE_COLS = load_model_and_cols()
except Exception as e:
    st.error(f"Error loading model/resources:\n{e}")
    st.stop()

# -----------------------------
# Biology helper functions (same as notebook)
# -----------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
hydropathy = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

def aa_composition(seq):
    seq = seq.strip().upper()
    counts = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
            total += 1
    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}

def compute_physchem(seq):
    seq = seq.strip().upper()
    length = len(seq)
    if length == 0:
        return {"seq_length": 0, "hydrophobicity": 0.0, "aromatic_fraction": 0.0}
    hyd = sum(hydropathy.get(aa,0) for aa in seq) / length
    aromatic = sum(seq.count(aa) for aa in "FWY") / length
    return {"seq_length": length, "hydrophobicity": hyd, "aromatic_fraction": aromatic}

def build_feature_vector(seq):
    aa_feats = aa_composition(seq)
    phys = compute_physchem(seq)
    row = []
    for c in FEATURE_COLS:
        if c in aa_feats:
            row.append(aa_feats[c])
        elif c in phys:
            row.append(phys[c])
        else:
            row.append(0.0)
    return np.array(row).reshape(1, -1)

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("", ["Predict", "Project Details", "Samples / Download"])

# -----------------------------
# Hero: 3D viewer (NGL) embedded via HTML
# -----------------------------
st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
st.markdown("### ")
st.markdown("###")
# embed viewer - if internet not available it will show error text inside container
pdbId = st.sidebar.text_input("Hero PDB (optional)", value="1EMA", help="Change PDB ID for background viewer")
ngl_html = f"""
<div id="viewport" style="width:100%; height:320px; border-radius:10px; overflow:hidden; background:#071029"></div>
<script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>
<script>
const stage = new NGL.Stage("viewport");
fetch("https://files.rcsb.org/download/{pdbId}.pdb")
  .then(resp => resp.text())
  .then(data => {{
    const blob = new Blob([data], {{type: "chemical/x-pdb"}});
    stage.loadFile(blob, {{ ext: "pdb" }}).then(o => {{
      o.addRepresentation("cartoon", {{ color: "chainname" }});
      o.addRepresentation("surface", {{ opacity: 0.12 }});
      stage.autoView();
      stage.setSpin(true);
      stage.spin(0.6);
    }});
  }})
  .catch(e => {{
    document.getElementById("viewport").innerText = "3D viewer failed to load (internet blocked?)";
  }});
</script>
"""
# render small hero only on Predict and Project Details
if page in ["Predict", "Project Details"]:
    try:
        html(ngl_html, height=320, scrolling=False)
    except Exception:
        st.info("3D viewer could not be embedded in this environment.")

# -----------------------------
# Page: Predict
# -----------------------------
if page == "Predict":
    st.header("Protein Solubility Predictor")
    st.write("Paste a protein sequence (single-letter amino acids). The model predicts Soluble vs Insoluble as observed in E. coli expression experiments.")

    # Input (use session state for prefill)
    seq_input = st.text_area("Sequence", value=st.session_state.get("seq_input", ""), height=200, placeholder="Paste sequence (no spaces/newlines preferred).")
    col1, col2 = st.columns([2,1])
    with col1:
        if st.button("Predict"):
            seq = seq_input.strip().replace("\n", "").replace(" ", "").upper()
            st.session_state["seq_input"] = seq  # store
            if len(seq) < 10:
                st.error("Please paste a valid protein sequence (at least 10 amino acids).")
            else:
                X = build_feature_vector(seq)
                proba = float(model.predict_proba(X)[0][1])
                pred = "Soluble" if proba >= 0.5 else "Insoluble"
                st.success(f"Prediction: {pred} — Probability (soluble): {proba:.3f}")
                # Feature snapshot
                fv = pd.Series(X.flatten(), index=FEATURE_COLS)
                st.markdown("**Top input features (snapshot)**")
                st.table(fv.sort_values(ascending=False).head(6))

                # Prepare downloadable CSV
                result_df = pd.DataFrame([{
                    "sequence": seq,
                    "prediction": pred,
                    "prob_soluble": proba,
                    "seq_length": len(seq)
                }])
                csv = result_df.to_csv(index=False).encode()
                st.download_button("Download prediction CSV", data=csv, file_name="prediction.csv", mime="text/csv")
    with col2:
        st.markdown("### Quick tips")
        st.write("- Short, charged proteins often express soluble in E. coli.")
        st.write("- Membrane proteins and cysteine-rich proteins are commonly insoluble without special protocols.")
        st.markdown("### Model info")
        st.write("- Model: XGBoost trained on UESolDS-derived features.")
        st.write("- Inputs: amino-acid composition (20), seq_length, hydrophobicity, aromatic fraction.")

# -----------------------------
# Page: Project Details
# -----------------------------
if page == "Project Details":
    st.header("Project Details & Results")
    st.write("**Goal:** Predict whether a protein expressed in *E. coli* will be soluble or insoluble using sequence-derived features.")
    st.subheader("Dataset")
    st.write("- UESolDS (curated E. coli protein solubility dataset). Train: ~70k sequences; balanced val/test (4k/4k).")
    st.subheader("Features used")
    st.write("- Amino-acid composition (20 features).")
    st.write("- Sequence length, hydrophobicity (Kyte-Doolittle avg), aromatic fraction (F/W/Y).")
    st.subheader("Modeling & Performance")
    st.write("- Baseline: Logistic Regression (AUC ~0.60).")
    st.write("- Random Forest (AUC ~0.69).")
    st.write("- Final: XGBoost + physchem (AUC ~0.73 on test).")
    st.subheader("Top features (from model)")
    st.write("- seq_length, C (cysteine), R (arginine), E (glutamate), N (asparagine), hydrophobicity.")
    # show ROC/confusion images if present
    if Path(ROC_PATH).exists():
        st.image(ROC_PATH, caption="ROC curve", use_column_width=True)
    if Path(CONF_PATH).exists():
        st.image(CONF_PATH, caption="Confusion matrix", use_column_width=True)

# -----------------------------
# Page: Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.header("Sample sequences & Download")
    st.write("Click a sample 'Use' button to prefill the input area on the Predict page.")
    samples = {
        "GFP (soluble)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
        "Thioredoxin (soluble)": "MKKIYFTKGQGPPAVPTTTGRSVPTIEVADKIVVGKPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
        "Bacteriorhodopsin (insoluble)": "MNGTEGPNFYVPFSNKTGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAFAVYMGVFSLAETNRFGAAHLP",
        "IL-2 (aggregation-prone)": "MYRMQLLSCIALSLALVTNSVTKTEANLAALEAKDSPQTHSLLEDAQQISLDKNQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLENELGALQR"
    }
    for name, seq in samples.items():
        cols = st.columns([3,1])
        cols[0].code(seq, language="text")
        if cols[1].button(f"Use {name}"):
            st.session_state["seq_input"] = seq
            st.rerun()

    st.markdown("---")
    st.write("Download model & feature columns (useful for reproducibility).")
    if Path(MODEL_PATH).exists():
        with open(MODEL_PATH, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="xgb2_solubility.joblib">Download model (joblib)</a>'
        st.markdown(href, unsafe_allow_html=True)
    if Path(FEATURE_COLS_PATH).exists():
        st.download_button("Download feature columns (.json)", data=open(FEATURE_COLS_PATH,"r").read(), file_name="feature_cols.json", mime="application/json")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ — UESolDS · XGBoost · Streamlit")
