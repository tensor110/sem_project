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
    "Attention Fusion": {
        "buildings": {"precision": 0.97, "recall": 0.96, "f1": 0.96, "support": 437},
        "forest":    {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 474},
        "glacier":   {"precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 553},
        "mountain":  {"precision": 0.93, "recall": 0.94, "f1": 0.93, "support": 525},
        "sea":       {"precision": 0.98, "recall": 0.97, "f1": 0.97, "support": 510},
        "street":    {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 501},
        "accuracy":  0.956,   # UPDATE with your actual number after training
    },
}

COLORS = {
    "Splice Fusion"   : "#8E44AD",
    "Weighted Fusion" : "#E67E22",
    "Attention Fusion": "#1ABC9C",
}
BASE_RESNET_ACC = 0.924   # image-only ResNet50 baseline

# ── 1. Fusion strategy explainer ──────────────────────────────────
st.subheader("⚙️ Fusion Strategies")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    #### Splice (Concatenation) Fusion
    ```
    Image feat → proj → 512 ─┐
                               ├→ concat(1024) → MLP → 6
    Text  feat → proj → 512 ─┘
    ```
    Blindly concatenates both features.
    Gives classifier full access to both
    independently.
    """)

with c2:
    st.markdown("""
    #### Weighted Fusion (δ = 0.6)
    ```
    Image feat → proj → 512 ─┐
                               ├→ δ·v+(1-δ)·t → 512 → MLP → 6
    Text  feat → proj → 512 ─┘
    ```
    Fixed-ratio mixing. δ=0.6 means
    image is trusted slightly more
    than text.
    """)

with c3:
    st.markdown("""
    #### Attention Fusion ✨
    ```
    Image feat → proj → Q (512) ─┐
                                   ├→ Q·Kᵀ/√d → sigmoid → weight
    Text  feat → proj → K,V(512)─┘
    fused = LayerNorm(Q + weight·V)
         → MLP → 6
    ```
    Dynamic per-sample weighting.
    Model learns when to trust text
    vs image for each input.
    """)

st.info("""
**Why attention is better:**  For a clear image (e.g. obvious forest), attention weight → low
(image alone is enough). For an ambiguous image, attention weight → high (text helps resolve it).
The model adapts per sample instead of using a fixed strategy.
""")

st.markdown("---")

# ── 2. Accuracy comparison ────────────────────────────────────────
st.subheader("📊 Accuracy: Image-Only vs All Fusion Strategies")

models_all = [
    "ResNet50\n(image only)",
    "Splice Fusion\n(image+text)",
    "Weighted Fusion\n(image+text)",
    "Attention Fusion\n(image+text)",
]
accs_all = [
    BASE_RESNET_ACC * 100,
    FUSION_REPORT["Splice Fusion"]["accuracy"] * 100,
    FUSION_REPORT["Weighted Fusion"]["accuracy"] * 100,
    FUSION_REPORT["Attention Fusion"]["accuracy"] * 100,
]
colors_all = ["#2B5BA8", "#8E44AD", "#E67E22", "#1ABC9C"]

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
    height=420,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
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
col4.metric(
    "Attention Fusion ✨",
    f"{FUSION_REPORT['Attention Fusion']['accuracy']*100:.1f}%",
    f"+{(FUSION_REPORT['Attention Fusion']['accuracy']-BASE_RESNET_ACC)*100:.1f}%"
)

st.markdown("---")

# ── 3. Per-class F1 comparison ────────────────────────────────────
st.subheader("📋 Per-class F1 — All Fusion Strategies")

rows = []
for cls in CLASS_NAMES:
    rows.append({
        "Class"         : cls.capitalize(),
        "Splice F1"     : FUSION_REPORT["Splice Fusion"][cls]["f1"],
        "Weighted F1"   : FUSION_REPORT["Weighted Fusion"][cls]["f1"],
        "Attention F1"  : FUSION_REPORT["Attention Fusion"][cls]["f1"],
    })
df = pd.DataFrame(rows)

fig2 = go.Figure()
fig2.add_trace(go.Bar(name="Splice",    x=df["Class"], y=df["Splice F1"],    marker_color="#8E44AD"))
fig2.add_trace(go.Bar(name="Weighted",  x=df["Class"], y=df["Weighted F1"],  marker_color="#E67E22"))
fig2.add_trace(go.Bar(name="Attention", x=df["Class"], y=df["Attention F1"], marker_color="#1ABC9C"))
fig2.update_layout(
    barmode="group",
    yaxis=dict(range=[0.8, 1.02], title="F1 Score"),
    height=380,
    margin=dict(t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 5. Attention weight visualization ────────────────────────────
st.subheader("🔍 Attention Weight Analysis")
st.markdown("""
The attention model learns a **per-sample weight (0→1)** that controls how much the text
description contributes to the final prediction.

- Weight **→ 1**: model heavily relied on text (image was ambiguous or hard)
- Weight **→ 0**: model relied mostly on image (image was clear enough alone)

Upload the attention weight plot to show this below:
""")

attn_png = st.file_uploader(
    "Attention weights per class PNG (attention_weights_per_class.png)",
    type=["png", "jpg"], key="attn_weights"
)
if attn_png:
    st.image(attn_png, caption="Mean Attention Weight per Class", use_container_width=True)
else:
    # Placeholder bar chart using placeholder values
    import plotly.express as px
    placeholder = {
        "buildings": 0.62, "forest": 0.41, "glacier": 0.71,
        "mountain": 0.68, "sea": 0.45, "street": 0.58
    }
    fig_attn = go.Figure(go.Bar(
        x=list(placeholder.keys()),
        y=list(placeholder.values()),
        marker_color=["#1ABC9C" if v >= 0.5 else "#E67E22" for v in placeholder.values()],
        text=[f"{v:.2f}" for v in placeholder.values()],
        textposition="outside",
    ))
    fig_attn.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       annotation_text="Threshold 0.5")
    fig_attn.update_layout(
        title="Mean Attention Weight per Class (placeholder — replace with your trained values)",
        yaxis=dict(range=[0, 1.15], title="Attention Weight"),
        height=360, margin=dict(t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_attn, use_container_width=True)
    st.caption("This is placeholder data. Upload your actual PNG after training.")
st.subheader("📈 Training Curves & Confusion Matrices")
col1, col2, col3 = st.columns(3)
with col1:
    curves = st.file_uploader(
        "All fusion curves (all_fusion_comparison.png)",
        type=["png", "jpg"], key="fusion_curves"
    )
    if curves:
        st.image(curves, caption="All Fusion — Accuracy Curves", use_container_width=True)
with col2:
    attn_curves = st.file_uploader(
        "Attention curves (attention_fusion_curves.png)",
        type=["png", "jpg"], key="attn_curves"
    )
    if attn_curves:
        st.image(attn_curves, caption="Attention Fusion Training", use_container_width=True)
with col3:
    cms = st.file_uploader(
        "All confusion matrices (all_fusion_confusion_matrices.png)",
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
