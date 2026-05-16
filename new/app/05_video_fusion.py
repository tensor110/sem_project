import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.title("🔀 Video Trimodal Fusion")
st.markdown("Fusing **Video (CNN+LSTM)** + **Text (DistilBERT)** + **Audio (gTTS→MFCC)** for action recognition")
st.markdown("---")

CLASS_NAMES = [
    "Basketball", "Biking", "Bowling", "CliffDiving",
    "GolfSwing", "HorseRiding", "Skiing", "Surfing",
    "TennisSwing", "SkateBoarding"
]

# ── UPDATE these with your actual trimodal numbers ────────────────
TRIMODAL_REPORT = {
    "Basketball"  : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 27},
    "Biking"      : {"precision": 1.00, "recall": 0.97, "f1": 0.98, "support": 29},
    "Bowling"     : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 23},
    "CliffDiving" : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 25},
    "GolfSwing"   : {"precision": 0.93, "recall": 0.95, "f1": 0.94, "support": 21},
    "HorseRiding" : {"precision": 1.00, "recall": 0.97, "f1": 0.98, "support": 31},
    "Skiing"      : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 27},
    "Surfing"     : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 28},
    "TennisSwing" : {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 30},
    "SkateBoarding": {"precision": 0.96, "recall": 0.96, "f1": 0.96, "support": 25},
    "accuracy"    : 0.981,
}

CNN_LSTM_ACC  = 0.9737
TRIMODAL_ACC  = TRIMODAL_REPORT["accuracy"]

# ── 1. Architecture ───────────────────────────────────────────────
st.subheader("🏗️ Trimodal Fusion Architecture")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    **🎬 Visual Branch**
    ```
    16 frames
       ↓
    ResNet50 CNN
    (layer4 unfrozen)
       ↓
    (B×T, 2048)
       ↓
    2-layer LSTM
       ↓
    Last state (512)
       ↓
    Linear → 256
    ```
    """)
with c2:
    st.markdown("""
    **💬 Text Branch**
    ```
    Generated caption
    (templates + word pools)
       ↓
    DistilBERT (frozen)
       ↓
    [CLS] token (768)
       ↓
    Linear → 256
    ```
    """)
with c3:
    st.markdown("""
    **🔊 Audio Branch**
    ```
    Same caption text
       ↓
    gTTS → MP3 audio
       ↓
    librosa MFCC (40 coeff)
       ↓
    mean + std + max
    = 120-d vector
       ↓
    MLP → 256
    ```
    """)

st.markdown("```\n256 + 256 + 256 = 768  →  Linear(512) → ReLU → Linear(256) → ReLU → Linear(10)\n```")
st.markdown("---")

# ── 2. Accuracy comparison ────────────────────────────────────────
st.subheader("📊 CNN+LSTM vs Trimodal Fusion")

models = ["CNN+LSTM\n(video only)", "Trimodal Fusion\n(video+text+audio)"]
accs   = [CNN_LSTM_ACC * 100, TRIMODAL_ACC * 100]

fig = go.Figure(go.Bar(
    x=models, y=accs,
    marker_color=["#27AE60", "#8E44AD"],
    text=[f"{a:.2f}%" for a in accs],
    textposition="outside",
    width=0.35,
))
fig.update_layout(
    yaxis=dict(range=[95, 100.5], title="Accuracy (%)"),
    height=380,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("CNN+LSTM (baseline)", f"{CNN_LSTM_ACC*100:.2f}%")
col2.metric(
    "Trimodal Fusion",
    f"{TRIMODAL_ACC*100:.2f}%",
    f"+{(TRIMODAL_ACC - CNN_LSTM_ACC)*100:.2f}%"
)
col3.metric("Improvement", f"+{(TRIMODAL_ACC - CNN_LSTM_ACC)*100:.2f}%")

st.markdown("---")

# ── 3. Per-class comparison bar ───────────────────────────────────
st.subheader("📋 Per-class F1 Comparison")

CNN_LSTM_F1 = [1.00, 0.98, 1.00, 1.00, 0.93, 0.97, 1.00, 1.00, 0.92, 0.94]
TRIMODAL_F1 = [TRIMODAL_REPORT[c]["f1"] for c in CLASS_NAMES]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="CNN+LSTM",
    x=CLASS_NAMES, y=CNN_LSTM_F1,
    marker_color="#27AE60",
))
fig2.add_trace(go.Bar(
    name="Trimodal Fusion",
    x=CLASS_NAMES, y=TRIMODAL_F1,
    marker_color="#8E44AD",
))
fig2.update_layout(
    barmode="group",
    yaxis=dict(range=[0.88, 1.04], title="F1 Score"),
    height=380,
    margin=dict(t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis_tickangle=-30,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 4. Classification report table ───────────────────────────────
st.subheader("📄 Trimodal Classification Report")
rows = [{
    "Class"    : cls,
    "Precision": f"{TRIMODAL_REPORT[cls]['precision']:.2f}",
    "Recall"   : f"{TRIMODAL_REPORT[cls]['recall']:.2f}",
    "F1 Score" : f"{TRIMODAL_REPORT[cls]['f1']:.2f}",
    "Support"  : TRIMODAL_REPORT[cls]["support"],
} for cls in CLASS_NAMES]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{TRIMODAL_ACC*100:.2f}%")
m2.metric("Macro F1", "0.98")
m3.metric("Weighted F1", "0.98")

st.markdown("---")

# ── 5. PNG upload ─────────────────────────────────────────────────
st.subheader("📈 Training Curves & Confusion Matrix")
col1, col2 = st.columns(2)
with col1:
    png1 = st.file_uploader(
        "Upload trimodal training curves PNG",
        type=["png", "jpg"], key="tri_curves"
    )
    if png1:
        st.image(png1, use_container_width=True)
with col2:
    png2 = st.file_uploader(
        "Upload trimodal confusion matrix PNG",
        type=["png", "jpg"], key="tri_cm"
    )
    if png2:
        st.image(png2, use_container_width=True)

st.markdown("---")

# ── 6. Audio pipeline explanation ─────────────────────────────────
st.subheader("🔊 Audio Feature Pipeline")
st.markdown("""
| Step | Operation | Output |
|---|---|---|
| 1 | Generate natural-language action caption | String |
| 2 | `gTTS` → convert text to spoken MP3 | Audio file |
| 3 | `librosa.load()` → waveform at 22050 Hz | Numpy array |
| 4 | `librosa.feature.mfcc()` → 40 MFCC coefficients | (40, T) |
| 5 | Pad/truncate to T=128 timesteps | (40, 128) |
| 6 | Aggregate: mean + std + max across time | (120,) |
| 7 | Audio MLP: 120 → 256 → 256 | (256,) |

The same caption text drives both the **text branch** (semantics via DistilBERT)
and the **audio branch** (acoustic patterns via MFCC). Different seeds ensure the two
branches see varied descriptions per sample, preventing shortcut learning.
""")
