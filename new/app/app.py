import streamlit as st
import importlib.util
import os

st.set_page_config(
    page_title="Deep Learning — News Content Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Absolute path to the folder this file lives in — works on any machine / cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png", width=72
)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Overview"              : "01_overview.py",
    "🖼️  Image Models"          : "02_image_models.py",
    "🔀  Image Fusion"          : "03_image_fusion.py",
    "🎬  Video Model"           : "04_video_model.py",
    "🔀  Video Trimodal Fusion" : "05_video_fusion.py",
    "🔮  Live Demo"             : "06_live_demo.py",
}

page = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Deep Learning · News Content Recognition\nFinal Year Project · 2024")

# ── Load selected page using absolute path ────────────────────────
def load_page(filename):
    path   = os.path.join(BASE_DIR, "pages", filename)
    spec   = importlib.util.spec_from_file_location("page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

load_page(PAGES[page])