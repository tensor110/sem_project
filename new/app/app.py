import streamlit as st

st.set_page_config(
    page_title="Deep Learning — News Content Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png", width=72
)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Overview":              "pages/01_overview.py",
    "🖼️  Image Models":          "pages/02_image_models.py",
    "🔀  Image Fusion":          "pages/03_image_fusion.py",
    "🎬  Video Model":           "pages/04_video_model.py",
    "🔀  Video Trimodal Fusion": "pages/05_video_fusion.py",
    "🔮  Live Demo":             "pages/06_live_demo.py",
}

page = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Deep Learning · News Content Recognition\nFinal Year Project · 2024")

# ── Route to page ─────────────────────────────────────────────────
import importlib.util, sys, os

def load_page(path):
    spec   = importlib.util.spec_from_file_location("page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

load_page(PAGES[page])
