import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.title("🔀 Image Multimodal Fusion")
st.markdown("Combining **ResNet50 image features** + **DistilBERT text features** using two fusion strategies")
st.markdown("---")

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ── Classification report data ────────────────────────────────────
# UPDATE THESE with your actual numbers after training
FUSION_REPORT = {
    "Splice Fusion": {
        "buildings": {"precision": 0.96, "recall": 0.95, "f1": 0.95, "support": 437},
        "forest":    {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 474},
        "glacier":   {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 553},
        "mountain":  {"precision": 0.92, "recall": 0.93, "f1": 0.92, "support": 525},
        "sea":       {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 510},
        "street":    {"precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 501},
        "accuracy":  0.945,
    },
    "Weighted Fusion": {
        "buildings": {"precision": 0.95, "recall": 0.94, "f1": 0.94, "support": 437},
        "forest":    {"precision": 0.99, "recall": 0.98, "f1": 0.98, "support": 474},
        "glacier":   {"precision": 0.92, "recall": 0.92, "f1": 0.92, "support": 553},
        "mountain":  {"precision": 0.91, "recall": 0.92, "f1": 0.91, "support": 525},
        "sea":       {"precision": 0.97, "recall": 0.96, "f1": 0.96, "support": 510},
        "street":    {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 501},
        "accuracy":  0.938,
    },
}

COLORS = {"Splice Fusion": "#8E44AD", "Weighted Fusion": "#E67E22"}
BASE_RESNET_ACC = 0.924   # image-only ResNet50 baseline

# ── 1. Fusion strategy explainer ──────────────────────────────────
st.subheader("⚙️ Fusion Strategies")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    #### Splice (Concatenation) Fusion
    ```
    Image feat (2048) → proj → 512-d ─┐
                                       ├→ concat(1024) → MLP → 6
    Text feat  (768)  → proj → 512-d ─┘
    ```
    Simply concatenates both modality projections.
    Gives the classifier full access to both feature spaces independently.
    """)

with c2:
    st.markdown("""
    #### Weighted Fusion (δ = 0.6)
    ```
    Image feat (2048) → proj → 512-d ─┐
                                       ├→ δ·v + (1-δ)·t → 512 → MLP → 6
    Text feat  (768)  → proj → 512-d ─┘
    ```
    Weighted sum of projections.
    δ=0.6 means image is trusted slightly more than text.
    """)

st.markdown("---")

# ── 2. Accuracy comparison ────────────────────────────────────────
st.subheader("📊 Accuracy: Image-Only vs Fusion")

models_all = ["ResNet50\n(image only)", "Splice Fusion\n(image+text)", "Weighted Fusion\n(image+text)"]
accs_all   = [
    BASE_RESNET_ACC * 100,
    FUSION_REPORT["Splice Fusion"]["accuracy"] * 100,
    FUSION_REPORT["Weighted Fusion"]["accuracy"] * 100,
]
colors_all = ["#2B5BA8", "#8E44AD", "#E67E22"]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=models_all, y=accs_all,
    marker_color=colors_all,
    text=[f"{a:.1f}%" for a in accs_all],
    textposition="outside",
    width=0.45,
))
fig.update_layout(
    yaxis=dict(range=[88, 100], title="Accuracy (%)"),
    height=400,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# improvement callouts
col1, col2, col3 = st.columns(3)
col1.metric("ResNet50 (baseline)", f"{BASE_RESNET_ACC*100:.1f}%")
col2.metric(
    "Splice Fusion",
    f"{FUSION_REPORT['Splice Fusion']['accuracy']*100:.1f}%",
    f"+{(FUSION_REPORT['Splice Fusion']['accuracy']-BASE_RESNET_ACC)*100:.1f}%"
)
col3.metric(
    "Weighted Fusion",
    f"{FUSION_REPORT['Weighted Fusion']['accuracy']*100:.1f}%",
    f"+{(FUSION_REPORT['Weighted Fusion']['accuracy']-BASE_RESNET_ACC)*100:.1f}%"
)

st.markdown("---")

# ── 3. Per-class F1 comparison ────────────────────────────────────
st.subheader("📋 Per-class F1 — Both Fusion Strategies")

rows = []
for cls in CLASS_NAMES:
    rows.append({
        "Class"         : cls.capitalize(),
        "Splice F1"     : FUSION_REPORT["Splice Fusion"][cls]["f1"],
        "Weighted F1"   : FUSION_REPORT["Weighted Fusion"][cls]["f1"],
        "Splice Prec"   : FUSION_REPORT["Splice Fusion"][cls]["precision"],
        "Weighted Prec" : FUSION_REPORT["Weighted Fusion"][cls]["precision"],
    })
df = pd.DataFrame(rows)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="Splice Fusion",
    x=df["Class"], y=df["Splice F1"],
    marker_color="#8E44AD",
))
fig2.add_trace(go.Bar(
    name="Weighted Fusion",
    x=df["Class"], y=df["Weighted F1"],
    marker_color="#E67E22",
))
fig2.update_layout(
    barmode="group",
    yaxis=dict(range=[0.8, 1.02], title="F1 Score"),
    height=350,
    margin=dict(t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 4. Saved PNG plots ────────────────────────────────────────────
st.subheader("📈 Training Curves & Confusion Matrices")
col1, col2 = st.columns(2)
with col1:
    curves = st.file_uploader(
        "Fusion training curves PNG (fusion_curves.png)",
        type=["png", "jpg"], key="fusion_curves"
    )
    if curves:
        st.image(curves, caption="Fusion Training Curves", use_container_width=True)
with col2:
    cms = st.file_uploader(
        "Fusion confusion matrices PNG (fusion_confusion_matrices.png)",
        type=["png", "jpg"], key="fusion_cm"
    )
    if cms:
        st.image(cms, caption="Fusion Confusion Matrices", use_container_width=True)

st.markdown("---")

# ── 5. Word pool explanation ──────────────────────────────────────
st.subheader("💬 How Text Features Are Generated")
st.markdown("""
Since the Intel Image dataset has no captions, text descriptions are **synthetically generated**
from class-specific word pools:

- **3 words** from the correct class pool
- **2 words** from a random other class (adds noise → prevents text from trivially solving the task)
- **1 shared ambiguous word**

This forces the model to genuinely fuse both modalities rather than relying on text alone.
""")

example_col1, example_col2 = st.columns(2)
with example_col1:
    st.markdown("**Example — class: `forest`**")
    st.code('"dense wild canopy urban light outdoor"')
    st.caption("3 forest words + 2 urban words + 1 shared word → shuffled")
with example_col2:
    st.markdown("**Example — class: `sea`**")
    st.code('"ocean horizon rocky outdoor coastal road"')
    st.caption("3 sea words + 2 mountain words + 1 shared word → shuffled")
