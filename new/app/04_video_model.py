import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.title("🎬 Video Model — CNN + LSTM")
st.markdown("ResNet50 (layer4 unfrozen) + 2-layer LSTM trained on UCF-10 dataset (10 action classes)")
st.markdown("---")

CLASS_NAMES = [
    "Basketball", "Biking", "Bowling", "CliffDiving",
    "GolfSwing", "HorseRiding", "Skiing", "Surfing",
    "TennisSwing", "SkateBoarding"
]

# ── Actual results from your trained notebook ─────────────────────
REPORT = {
    "Basketball"  : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 27},
    "Biking"      : {"precision": 1.00, "recall": 0.97, "f1": 0.98, "support": 29},
    "Bowling"     : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 23},
    "CliffDiving" : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 25},
    "GolfSwing"   : {"precision": 0.91, "recall": 0.95, "f1": 0.93, "support": 21},
    "HorseRiding" : {"precision": 1.00, "recall": 0.94, "f1": 0.97, "support": 31},
    "Skiing"      : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 27},
    "Surfing"     : {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 28},
    "TennisSwing" : {"precision": 0.90, "recall": 0.93, "f1": 0.92, "support": 30},
    "SkateBoarding": {"precision": 0.92, "recall": 0.96, "f1": 0.94, "support": 25},
    "accuracy"    : 0.9737,
}

# ── 1. Architecture diagram ───────────────────────────────────────
st.subheader("🏗️ CNN + LSTM Architecture")
st.markdown("""
```
Input video (8 videos × 16 frames × 3 × 224 × 224)
        ↓
Reshape → (128, 3, 224, 224)   [all frames treated as independent images]
        ↓
ResNet50 CNN  (layer4 unfrozen — fine-tunes high-level spatial features)
        ↓
CNN output: (128, 2048, 1, 1)
        ↓
Reshape → (8, 16, 2048)        [restore batch × time × features]
        ↓
2-layer LSTM  hidden_size=512, dropout=0.3
        ↓
Last frame hidden state: (8, 512)
        ↓
Classifier: 512 → 256 → ReLU → Dropout(0.4) → 10 classes
```
""")

st.markdown("---")

# ── 2. Training results ───────────────────────────────────────────
st.subheader("📊 Training Results")

# Epoch-level data from your actual training output
epochs     = list(range(1, 11))
train_acc  = [0.3917, 0.6497, 0.7674, 0.7815, 0.8409, 0.9058, 0.9379, 0.9435, 0.9426, 0.9331]
val_acc    = [0.6729, 0.8797, 0.8008, 0.8947, 0.8647, 0.9549, 0.9624, 0.9624, 0.9624, 0.9737]
train_loss = [1.8624, 1.0489, 0.7283, 0.6686, 0.4784, 0.3404, 0.2324, 0.2225, 0.2058, 0.2163]
val_loss   = [1.0408, 0.5492, 0.6375, 0.3293, 0.4230, 0.1682, 0.1358, 0.1316, 0.1279, 0.1196]

col1, col2 = st.columns(2)

with col1:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=train_acc, name="Train",
        mode="lines+markers", line=dict(color="steelblue", width=2),
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=val_acc, name="Val",
        mode="lines+markers", line=dict(color="darkorange", width=2),
    ))
    fig_acc.update_layout(
        title="Accuracy per Epoch",
        xaxis_title="Epoch", yaxis_title="Accuracy",
        yaxis=dict(range=[0.3, 1.0]),
        height=340, margin=dict(t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=train_loss, name="Train",
        mode="lines+markers", line=dict(color="steelblue", width=2),
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=val_loss, name="Val",
        mode="lines+markers", line=dict(color="darkorange", width=2),
    ))
    fig_loss.update_layout(
        title="Loss per Epoch",
        xaxis_title="Epoch", yaxis_title="Loss",
        height=340, margin=dict(t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

st.markdown("---")

# ── 3. Per-class results ──────────────────────────────────────────
st.subheader("📋 Per-class Classification Report")

rows = [{
    "Class"    : cls,
    "Precision": f"{REPORT[cls]['precision']:.2f}",
    "Recall"   : f"{REPORT[cls]['recall']:.2f}",
    "F1 Score" : f"{REPORT[cls]['f1']:.2f}",
    "Support"  : REPORT[cls]["support"],
} for cls in CLASS_NAMES]

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

m1, m2, m3 = st.columns(3)
m1.metric("Overall Accuracy", f"{REPORT['accuracy']*100:.2f}%")
m2.metric("Macro F1",  "0.97")
m3.metric("Weighted F1", "0.97")

st.markdown("---")

# ── 4. F1 bar chart per class ─────────────────────────────────────
st.subheader("📊 F1 Score per Class")
f1s = [REPORT[c]["f1"] for c in CLASS_NAMES]
fig3 = go.Figure(go.Bar(
    x=CLASS_NAMES, y=f1s,
    marker_color=["#27AE60" if f >= 0.95 else "#E67E22" if f >= 0.90 else "#E74C3C" for f in f1s],
    text=[f"{f:.2f}" for f in f1s],
    textposition="outside",
))
fig3.update_layout(
    yaxis=dict(range=[0.85, 1.05], title="F1 Score"),
    height=360, margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── 5. Confusion matrix upload ────────────────────────────────────
st.subheader("📈 Confusion Matrix")
cm_file = st.file_uploader(
    "Upload confusion matrix PNG (cnn_lstm_results.png)",
    type=["png", "jpg"], key="video_cm"
)
if cm_file:
    st.image(cm_file, caption="CNN+LSTM Confusion Matrix", use_container_width=True)

st.markdown("---")

# ── 6. Model hyperparameters ──────────────────────────────────────
st.subheader("⚙️ Hyperparameters")
hp = {
    "Frames per video"  : "16",
    "Batch size"        : "8",
    "Epochs"            : "10",
    "Optimizer"         : "Adam (lr=1e-4)",
    "Scheduler"         : "StepLR (step=5, γ=0.1)",
    "Loss"              : "CrossEntropyLoss",
    "LSTM hidden size"  : "512",
    "LSTM layers"       : "2",
    "LSTM dropout"      : "0.3",
    "ResNet layer4"     : "Unfrozen (fine-tuned)",
    "Train/Val split"   : "80 / 20",
    "Trainable params"  : "~7.48M",
}
df_hp = pd.DataFrame(list(hp.items()), columns=["Hyperparameter", "Value"])
st.dataframe(df_hp, use_container_width=True, hide_index=True)
