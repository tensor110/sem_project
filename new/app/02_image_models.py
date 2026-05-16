import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.title("🖼️ Image Models")
st.markdown("ResNet50 · VGG16 · Custom CNN trained on Intel Image Classification dataset (6 classes)")
st.markdown("---")

# ── Classification report data (paste your numbers here) ─────────
REPORT = {
    "ResNet50": {
        "buildings": {"precision": 0.95, "recall": 0.94, "f1": 0.94, "support": 437},
        "forest":    {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 474},
        "glacier":   {"precision": 0.92, "recall": 0.91, "f1": 0.91, "support": 553},
        "mountain":  {"precision": 0.90, "recall": 0.93, "f1": 0.91, "support": 525},
        "sea":       {"precision": 0.97, "recall": 0.96, "f1": 0.96, "support": 510},
        "street":    {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 501},
        "accuracy":  0.924,
    },
    "VGG16": {
        "buildings": {"precision": 0.93, "recall": 0.92, "f1": 0.92, "support": 437},
        "forest":    {"precision": 0.98, "recall": 0.97, "f1": 0.97, "support": 474},
        "glacier":   {"precision": 0.90, "recall": 0.89, "f1": 0.89, "support": 553},
        "mountain":  {"precision": 0.88, "recall": 0.90, "f1": 0.89, "support": 525},
        "sea":       {"precision": 0.95, "recall": 0.94, "f1": 0.95, "support": 510},
        "street":    {"precision": 0.91, "recall": 0.91, "f1": 0.91, "support": 501},
        "accuracy":  0.921,
    },
    "Custom CNN": {
        "buildings": {"precision": 0.90, "recall": 0.88, "f1": 0.89, "support": 437},
        "forest":    {"precision": 0.97, "recall": 0.96, "f1": 0.96, "support": 474},
        "glacier":   {"precision": 0.85, "recall": 0.84, "f1": 0.84, "support": 553},
        "mountain":  {"precision": 0.82, "recall": 0.85, "f1": 0.83, "support": 525},
        "sea":       {"precision": 0.92, "recall": 0.91, "f1": 0.91, "support": 510},
        "street":    {"precision": 0.87, "recall": 0.87, "f1": 0.87, "support": 501},
        "accuracy":  0.885,
    },
}

COLORS = {
    "ResNet50"  : "#2B5BA8",
    "VGG16"     : "#27AE60",
    "Custom CNN": "#E74C3C",
}

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ── 1. Accuracy comparison bar ────────────────────────────────────
st.subheader("📊 Validation Accuracy Comparison")

fig = go.Figure()
models = list(REPORT.keys())
accs   = [REPORT[m]["accuracy"] * 100 for m in models]

fig.add_trace(go.Bar(
    x=models, y=accs,
    marker_color=[COLORS[m] for m in models],
    text=[f"{a:.1f}%" for a in accs],
    textposition="outside",
    width=0.4,
))
fig.update_layout(
    yaxis=dict(range=[0, 105], title="Accuracy (%)"),
    xaxis_title="Model",
    height=380,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── 2. Per-class F1 heatmap ───────────────────────────────────────
st.subheader("📋 Per-class F1 Score")

f1_data = {m: [REPORT[m][c]["f1"] for c in CLASS_NAMES] for m in models}
df_f1   = pd.DataFrame(f1_data, index=CLASS_NAMES)

fig2 = px.imshow(
    df_f1.T,
    text_auto=".2f",
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(x="Class", y="Model", color="F1"),
    zmin=0.7, zmax=1.0,
)
fig2.update_layout(height=280, margin=dict(t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 3. Saved PNG plots ────────────────────────────────────────────
st.subheader("📈 Training Curves & Confusion Matrices")
st.info(
    "Upload your saved PNG files from Google Drive below. "
    "They will display instantly for the panel."
)

col1, col2 = st.columns(2)
with col1:
    curves = st.file_uploader(
        "Training curves PNG (model_comparison.png / training_curves.png)",
        type=["png", "jpg"],
        key="img_curves"
    )
    if curves:
        st.image(curves, caption="Training Curves", use_container_width=True)

with col2:
    cms = st.file_uploader(
        "Confusion matrices PNG (confusion_matrices.png)",
        type=["png", "jpg"],
        key="img_cm"
    )
    if cms:
        st.image(cms, caption="Confusion Matrices", use_container_width=True)

st.markdown("---")

# ── 4. Detailed classification report table ───────────────────────
st.subheader("📄 Detailed Classification Report")
selected_model = st.selectbox("Select model", models)

rows = []
for cls in CLASS_NAMES:
    r = REPORT[selected_model][cls]
    rows.append({
        "Class"    : cls.capitalize(),
        "Precision": f"{r['precision']:.2f}",
        "Recall"   : f"{r['recall']:.2f}",
        "F1 Score" : f"{r['f1']:.2f}",
        "Support"  : r["support"],
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.metric("Overall Accuracy", f"{REPORT[selected_model]['accuracy']*100:.1f}%")

st.markdown("---")

# ── 5. Model architecture summary ────────────────────────────────
st.subheader("🏗️ Model Architecture Summary")
arch_data = {
    "Model"            : ["ResNet50",     "VGG16",       "Custom CNN"],
    "Backbone"         : ["ResNet50",     "VGG16",       "Scratch"],
    "Pretrained"       : ["✅ Yes",       "✅ Yes",      "❌ No"],
    "Frozen Backbone"  : ["✅ Yes",       "✅ Yes",      "N/A"],
    "Head"             : ["2048→256→6",   "4096→256→6",  "4-block→512→128→6"],
    "Trainable Params" : ["~262K",        "~1.05M",      "~26M"],
    "Val Accuracy"     : ["92.4%",        "92.1%",       "88.5%"],
}
st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)
