import streamlit as st

st.title("🧠 Deep Learning Based Content Recognition")
st.subheader("for News Images and Videos")
st.markdown("---")

# ── Project summary ───────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### About This Project
    This project implements deep learning pipelines for recognising content in
    news images and videos. It explores three model families and two fusion
    strategies, comparing unimodal vs multimodal approaches.

    The work is based on the paper:
    > *Deep Learning Based Content Recognition for News Images and Videos*
    > — Xue Lin & Xuejiao Ren, AIDLNN 2024
    """)

with col2:
    st.markdown("""
    ### Quick Stats
    | | |
    |---|---|
    | Image classes | **6** |
    | Video classes | **10** |
    | Image models  | **5** |
    | Video models  | **2** |
    | Best image acc| **96.5%** |
    | Best video acc| **97.4%** |
    """)

st.markdown("---")

# ── Architecture cards ────────────────────────────────────────────
st.subheader("📐 System Architecture")
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🖼️ Image Pipeline")
    st.markdown("""
    ```
    Intel Image Dataset (6 classes)
            ↓
    ┌───────────────────────┐
    │  ResNet50  │  VGG16  │  Custom CNN
    └───────────────────────┘
            ↓
    Image Feature (2048-d)
            ↓
    Text Feature via DistilBERT (768-d)
            ↓
    ┌─────────────────────────────┐
    │  Splice Fusion │  Weighted Fusion  │
    └─────────────────────────────┘
            ↓
       Classification (6 classes)
    ```
    """)

with c2:
    st.markdown("#### 🎬 Video Pipeline")
    st.markdown("""
    ```
    UCF-10 Dataset (10 classes)
            ↓
    Frame Extraction (16 frames/video)
            ↓
    ResNet50 CNN per frame (2048-d)
            ↓
    LSTM Temporal Modeling (512-d)
            ↓
    Text (DistilBERT) + Audio (MFCC)
            ↓
    Trimodal Fusion (768-d concat)
            ↓
      Classification (10 classes)
    ```
    """)

st.markdown("---")

# ── Dataset info ──────────────────────────────────────────────────
st.subheader("📦 Datasets")
d1, d2 = st.columns(2)

with d1:
    st.info("""
    **Intel Image Classification**
    - 6 scene classes
    - ~14,000 training images
    - ~3,000 test images
    - Source: Kaggle
    - Classes: buildings, forest, glacier,
      mountain, sea, street
    """)

with d2:
    st.info("""
    **UCF-101 (10-class subset)**
    - 10 action classes
    - ~1,328 valid video clips
    - 16 frames extracted per video
    - Classes: Basketball, Biking, Bowling,
      CliffDiving, GolfSwing, HorseRiding,
      Skiing, Surfing, TennisSwing, SkateBoarding
    """)

st.markdown("---")
st.caption("Use the sidebar to navigate between sections.")
