"""
live_demo.py  ─  Single-file Streamlit live-prediction app
Supports: Image (ResNet50 / VGG16 / CustomCNN / SpliceFusion / WeightedFusion / AttentionFusion)
          Video (CNN+LSTM / Trimodal Fusion)
Run: streamlit run live_demo.py
"""

import math, os, random, tempfile

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Recognition — Live Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark base ── */
.stApp {
    background: #080c14;
    color: #e8eaf0;
}

/* ── Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0d1321 0%, #101828 100%);
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-title span { color: #3b9eff; }
.hero-sub {
    font-size: 0.95rem;
    color: #6b7fa3;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0d1321;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7fa3;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1a2744 !important;
    color: #3b9eff !important;
}

/* ── Cards ── */
.card {
    background: #0d1321;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #3b9eff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Prediction badge ── */
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: linear-gradient(135deg, #0d2d1a, #0a2010);
    border: 1px solid #1a5c33;
    border-radius: 10px;
    padding: 1rem 1.6rem;
    margin: 1rem 0;
    width: 100%;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #34d26e;
    letter-spacing: -0.5px;
}
.pred-conf {
    font-size: 0.95rem;
    color: #6bc890;
    margin-left: auto;
    font-weight: 500;
}

/* ── Model selector pills ── */
.model-pill {
    display: inline-block;
    background: #1a2744;
    border: 1px solid #2a3d60;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #7da8e0;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}

/* ── Step labels ── */
.step-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #3b9eff;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ── Divider ── */
.divider { border-top: 1px solid #1e2d45; margin: 1.5rem 0; }

/* ── Optional tag ── */
.opt-tag {
    display: inline-block;
    background: #1e2d10;
    border: 1px solid #3a5a20;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    color: #7ec34a;
    font-family: 'Space Mono', monospace;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* ── Streamlit overrides ── */
.stFileUploader > div { background: #0d1321 !important; border-color: #1e2d45 !important; border-radius: 10px !important; }
.stSelectbox > div > div { background: #0d1321 !important; border-color: #1e2d45 !important; border-radius: 8px !important; }
.stTextInput > div > div { background: #0d1321 !important; border-color: #1e2d45 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1a3d7c, #1e4d9a) !important;
    color: #e8eaf0 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stAlert { border-radius: 8px !important; }
.stSpinner > div { border-top-color: #3b9eff !important; }
label { color: #b0bcd4 !important; font-size: 0.88rem !important; }

/* ── Frames grid ── */
.frames-label {
    font-size: 0.78rem;
    color: #6b7fa3;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
VIDEO_CLASSES = [
    'Basketball', 'Biking', 'Bowling', 'CliffDiving',
    'GolfSwing', 'HorseRiding', 'Skiing', 'Surfing',
    'TennisSwing', 'SkateBoarding'
]
FRAMES_PER_VIDEO = 16
device = torch.device("cpu")

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS — mirror your training notebooks exactly
# ─────────────────────────────────────────────────────────────────────────────

# ── IMAGE UNIMODAL ────────────────────────────────────────────────────────────

def build_resnet50(num_classes=6):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return m


def build_vgg16(num_classes=6):
    m = models.vgg16(weights=None)
    m.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return m


class CustomCNN(nn.Module):
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )

    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = self._block(3, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)
        self.block4 = self._block(128, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block4(self.block3(self.block2(self.block1(x)))))


# ── IMAGE FUSION (take raw image + pre-extracted 2048-d image feat + text feat) ──

class SpliceFusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.visual_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3))
        self.text_proj   = nn.Sequential(nn.Linear(768,  512), nn.ReLU(), nn.Dropout(0.3))
        self.classifier  = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_feat, txt_feat):
        return self.classifier(torch.cat([self.visual_proj(img_feat), self.text_proj(txt_feat)], dim=1))


class WeightedFusionModel(nn.Module):
    def __init__(self, num_classes=6, delta=0.6):
        super().__init__()
        self.delta = delta
        self.visual_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3))
        self.text_proj   = nn.Sequential(nn.Linear(768,  512), nn.ReLU(), nn.Dropout(0.3))
        self.classifier  = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_feat, txt_feat):
        v = self.visual_proj(img_feat)
        t = self.text_proj(txt_feat)
        return self.classifier(self.delta * v + (1 - self.delta) * t)


class AttentionFusionModel(nn.Module):
    def __init__(self, num_classes=6, proj_dim=512):
        super().__init__()
        self.scale       = math.sqrt(proj_dim)
        self.visual_proj = nn.Sequential(nn.Linear(2048, proj_dim), nn.ReLU(), nn.Dropout(0.3))
        self.text_proj   = nn.Sequential(nn.Linear(768,  proj_dim), nn.ReLU(), nn.Dropout(0.3))
        self.layer_norm  = nn.LayerNorm(proj_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_feat, txt_feat):
        Q = self.visual_proj(img_feat)
        K = V = self.text_proj(txt_feat)
        attn_weight = torch.sigmoid(torch.sum(Q * K, dim=1, keepdim=True) / self.scale)
        fused = self.layer_norm(Q + attn_weight * V)
        return self.classifier(fused)


# ── VIDEO UNIMODAL ────────────────────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512, num_layers=2):
        super().__init__()
        resnet    = models.resnet50(weights=None)
        self.cnn  = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])


# ── VIDEO TRIMODAL ─────────────────────────────────────────────────────────────

class TrimodalFusionModel(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512, num_layers=2):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.cnn  = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.visual_proj = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3))
        self.text_proj   = nn.Sequential(nn.Linear(768, 256),  nn.ReLU(), nn.Dropout(0.3))
        self.audio_proj  = nn.Sequential(
            nn.Linear(120, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, video, text_feat, audio_feat):
        B, T, C, H, W = video.shape
        cnn_out = self.cnn(video.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        v = self.visual_proj(lstm_out[:, -1, :])
        if text_feat.dim() == 3:
            text_feat = text_feat.squeeze(1)
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        return self.fusion(torch.cat([v, t, a], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Resnet backbone feature extractor (for fusion image models)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_resnet_backbone():

    NUM_CLASSES = 6

    # Build SAME architecture as training
    resnet_full = models.resnet50(weights=None)

    resnet_full.fc = nn.Sequential(
        nn.Linear(resnet_full.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )

    # Load trained ResNet50 checkpoint
    resnet_ckpt_path = "ResNet50.pth"

    state = torch.load(resnet_ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    resnet_full.load_state_dict(state)

    # Remove classifier head
    backbone = nn.Sequential(*list(resnet_full.children())[:-1])

    backbone.eval()

    return backbone


def extract_image_features(img_pil):
    """PIL image → (1, 2048) tensor via ResNet backbone."""
    backbone = _get_resnet_backbone()
    t = IMG_TRANSFORM(img_pil).unsqueeze(0)
    with torch.no_grad():
        feat = backbone(t)  # (1, 2048, 1, 1)
    return feat.view(1, -1)  # (1, 2048)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: DistilBERT text features
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_bert():
    from transformers import DistilBertTokenizer, DistilBertModel
    tok   = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


def get_text_features(text: str):
    """text → (1, 768) tensor."""
    tok, bert = _load_bert()
    inputs = tok(text, return_tensors="pt", padding=True,
                 truncation=True, max_length=64)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:, 0, :]  # (1, 768)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Audio MFCC features (gTTS → librosa)
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_features(text: str, n_mfcc=40, max_len=128):
    """text → gTTS → MFCC → (120,) tensor. Returns zeros on error."""
    try:
        from gtts import gTTS
        import librosa
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        tts.save(tmp)
        audio, sr = librosa.load(tmp, sr=22050)
        os.unlink(tmp)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
        feat = np.concatenate([mfcc.mean(1), mfcc.std(1), mfcc.max(1)])
        return torch.FloatTensor(feat)
    except Exception as e:
        st.warning(f"Audio extraction failed ({e}), using zeros.")
        return torch.zeros(n_mfcc * 3)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model, ckpt_file):
    """Load state dict from uploaded file object into model. Returns model or raises."""
    data = ckpt_file.read()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        f.write(data)
        tmp = f.name
    state = torch.load(tmp, map_location=device)
    os.unlink(tmp)
    # Handle nested dicts (model_state_dict key)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Video frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_bytes, num_frames=16):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    cap   = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.unlink(tmp)
    if total == 0:
        cap.release()
        return None
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int))
    frames, fi = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi in indices:
            frame = cv2.resize(frame, (224, 224))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fi += 1
    cap.release()
    return frames if len(frames) == num_frames else None


def frames_to_tensor(frames):
    return torch.stack([IMG_TRANSFORM(Image.fromarray(f)) for f in frames]).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Confidence bar chart
# ─────────────────────────────────────────────────────────────────────────────

def confidence_chart(probs, class_names):
    top_idx = int(np.argmax(probs))
    colors  = ["#3b9eff" if i == top_idx else "#1e2d45" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=class_names,
        orientation="h",
        marker_color=colors,
        marker_line_color=["#3b9eff" if i == top_idx else "#2a3d60" for i in range(len(probs))],
        marker_line_width=1,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#b0bcd4", size=12, family="DM Sans"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 120], showgrid=False, showticklabels=False,
                   title="", zeroline=False),
        yaxis=dict(tickfont=dict(color="#b0bcd4", size=13, family="DM Sans")),
        height=max(280, len(class_names) * 36),
        margin=dict(t=10, l=10, r=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">🔮 <span>Live</span> Prediction</div>
  <div class="hero-sub">Deep Learning · News Content Recognition · Upload → Predict</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["🖼️  Image Classification", "🎬  Video Action Recognition"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:

    st.markdown("""
    <div class="card">
      <div class="card-title">Scene Classes</div>
      buildings &nbsp;·&nbsp; forest &nbsp;·&nbsp; glacier &nbsp;·&nbsp;
      mountain &nbsp;·&nbsp; sea &nbsp;·&nbsp; street
    </div>
    """, unsafe_allow_html=True)

    img_col_left, img_col_right = st.columns([1, 1], gap="large")

    with img_col_left:

        # ── Step 1: Model ─────────────────────────────────────────
        st.markdown('<div class="step-label">① Model</div>', unsafe_allow_html=True)
        img_model_name = st.selectbox(
            "Select model architecture",
            ["ResNet50", "VGG16", "Custom CNN",
             "Splice Fusion (image + text)",
             "Weighted Fusion (image + text)",
             "Attention Fusion (image + text)"],
            label_visibility="collapsed",
            key="img_model_sel"
        )
        is_fusion_img = "Fusion" in img_model_name

        # ── Step 2: Checkpoint ────────────────────────────────────
        st.markdown('<div class="step-label" style="margin-top:1rem">② Checkpoint (.pth)</div>', unsafe_allow_html=True)
        img_ckpt = st.file_uploader(
            "Upload checkpoint", type=["pth"], key="img_ckpt",
            label_visibility="collapsed"
        )

        # ── Step 3: Image ─────────────────────────────────────────
        st.markdown('<div class="step-label" style="margin-top:1rem">③ Image</div>', unsafe_allow_html=True)
        img_file = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png"], key="img_file",
            label_visibility="collapsed"
        )

        # ── Step 4 (optional): Text description ──────────────────
        if is_fusion_img:
            st.markdown(
                '<div class="step-label" style="margin-top:1rem">④ Text Description '
                '<span class="opt-tag">OPTIONAL</span></div>',
                unsafe_allow_html=True
            )
            img_text_input = st.text_area(
                "Text description",
                placeholder="e.g.  dense green forest trees canopy outdoor\n"
                            "(leave blank to auto-generate from class word pools)",
                height=80, key="img_text_input",
                label_visibility="collapsed"
            )
            st.caption("DistilBERT encodes this into a 768-d embedding for fusion.")
        else:
            img_text_input = None

    with img_col_right:
        if img_file:
            img_pil = Image.open(img_file).convert("RGB")
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    # ── Inference ──────────────────────────────────────────────────────────
    if img_file:
        if not img_ckpt:
            st.info("⬆️  Upload the model checkpoint (.pth) to run prediction.")
        else:
            with st.spinner("Loading model · running inference…"):
                try:
                    img_pil = Image.open(img_file).convert("RGB")
                    nc      = len(IMAGE_CLASSES)

                    if img_model_name == "ResNet50":
                        model = build_resnet50(nc)
                        model = load_checkpoint(model, img_ckpt)
                        tensor = IMG_TRANSFORM(img_pil).unsqueeze(0)
                        with torch.no_grad():
                            probs = torch.softmax(model(tensor), dim=1)[0].numpy()

                    elif img_model_name == "VGG16":
                        model = build_vgg16(nc)
                        model = load_checkpoint(model, img_ckpt)
                        tensor = IMG_TRANSFORM(img_pil).unsqueeze(0)
                        with torch.no_grad():
                            probs = torch.softmax(model(tensor), dim=1)[0].numpy()

                    elif img_model_name == "Custom CNN":
                        model = CustomCNN(nc)
                        model = load_checkpoint(model, img_ckpt)
                        tensor = IMG_TRANSFORM(img_pil).unsqueeze(0)
                        with torch.no_grad():
                            probs = torch.softmax(model(tensor), dim=1)[0].numpy()

                    else:
                        # ── Fusion models need image features + text features ──
                        img_feat = extract_image_features(img_pil)

                        if img_text_input and img_text_input.strip():
                            txt_feat = get_text_features(img_text_input.strip())
                        else:
                            # Auto-generate a noisy word-pool description for inference
                            WORD_POOLS = {
                                'buildings': ['tall','urban','building','architecture','city','skyscraper','concrete'],
                                'forest':    ['dense','green','trees','forest','woodland','leaves','canopy'],
                                'glacier':   ['ice','cold','frozen','glacier','snow','arctic','frost'],
                                'mountain':  ['rocky','mountain','peak','summit','cliff','highland','steep'],
                                'sea':       ['ocean','water','waves','beach','coastal','shore','horizon'],
                                'street':    ['road','street','traffic','pavement','sidewalk','intersection','lane'],
                            }
                            SHARED = ['outdoor','light','natural','open','landscape','scene','environment']
                            cls  = random.choice(IMAGE_CLASSES)
                            own  = random.sample(WORD_POOLS[cls], 3)
                            other_cls = random.choice([c for c in IMAGE_CLASSES if c != cls])
                            other = random.sample(WORD_POOLS[other_cls], 2)
                            words = own + other + random.sample(SHARED, 1)
                            random.shuffle(words)
                            auto_text = " ".join(words)
                            st.caption(f"Auto-generated text: *\"{auto_text}\"*")
                            txt_feat = get_text_features(auto_text)

                        if "Splice" in img_model_name:
                            model = SpliceFusionModel(nc)
                        elif "Weighted" in img_model_name:
                            model = WeightedFusionModel(nc)
                        else:
                            model = AttentionFusionModel(nc)

                        model = load_checkpoint(model, img_ckpt)
                        with torch.no_grad():
                            probs = torch.softmax(model(img_feat, txt_feat), dim=1)[0].numpy()

                    pred_class  = IMAGE_CLASSES[int(np.argmax(probs))]
                    confidence  = float(np.max(probs))

                    st.markdown(f"""
                    <div class="pred-badge">
                      <span>🎯</span>
                      <span class="pred-label">{pred_class.upper()}</span>
                      <span class="pred-conf">{confidence*100:.1f}% confidence</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.plotly_chart(
                        confidence_chart(probs, [c.capitalize() for c in IMAGE_CLASSES]),
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.info("Make sure the checkpoint matches the selected model architecture.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:

    st.markdown("""
    <div class="card">
      <div class="card-title">Action Classes</div>
      Basketball &nbsp;·&nbsp; Biking &nbsp;·&nbsp; Bowling &nbsp;·&nbsp;
      CliffDiving &nbsp;·&nbsp; GolfSwing &nbsp;·&nbsp; HorseRiding &nbsp;·&nbsp;
      Skiing &nbsp;·&nbsp; Surfing &nbsp;·&nbsp; TennisSwing &nbsp;·&nbsp; SkateBoarding
    </div>
    """, unsafe_allow_html=True)

    vid_col_left, vid_col_right = st.columns([1, 1], gap="large")

    with vid_col_left:

        # ── Step 1: Model ─────────────────────────────────────────
        st.markdown('<div class="step-label">① Model</div>', unsafe_allow_html=True)
        vid_model_name = st.selectbox(
            "Select video model",
            ["CNN + LSTM (video only)",
             "Trimodal Fusion (video + text + audio)"],
            label_visibility="collapsed",
            key="vid_model_sel"
        )
        is_trimodal = "Trimodal" in vid_model_name

        # ── Step 2: Checkpoint ────────────────────────────────────
        st.markdown('<div class="step-label" style="margin-top:1rem">② Checkpoint (.pth)</div>', unsafe_allow_html=True)
        vid_ckpt = st.file_uploader(
            "Upload checkpoint", type=["pth"], key="vid_ckpt",
            label_visibility="collapsed"
        )

        # ── Step 3: Video ─────────────────────────────────────────
        st.markdown('<div class="step-label" style="margin-top:1rem">③ Video Clip</div>', unsafe_allow_html=True)
        vid_file = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov"], key="vid_file",
            label_visibility="collapsed"
        )

        if is_trimodal:
            # ── Step 4: Text (optional) ───────────────────────────
            st.markdown(
                '<div class="step-label" style="margin-top:1rem">④ Action Caption '
                '<span class="opt-tag">OPTIONAL</span></div>',
                unsafe_allow_html=True
            )
            vid_text_input = st.text_area(
                "Action caption",
                placeholder="e.g.  this is competitive footage of basketball players dribbling\n"
                            "(leave blank to auto-generate from action templates)",
                height=80, key="vid_text_input",
                label_visibility="collapsed"
            )
            st.caption("Text → DistilBERT (768-d) · Audio synthesised from same text via gTTS → MFCC (120-d)")
        else:
            vid_text_input = None

    with vid_col_right:
        if vid_file:
            st.video(vid_file)

    # ── Inference ──────────────────────────────────────────────────────────
    if vid_file:
        if not vid_ckpt:
            st.info("⬆️  Upload the model checkpoint (.pth) to run prediction.")
        else:
            with st.spinner("Extracting frames · loading model · running inference…"):
                try:
                    frames = extract_frames(vid_file.read())
                    if frames is None:
                        st.error("Could not extract 16 frames. Try a longer video clip (>2 s).")
                    else:
                        # Show sampled frames
                        st.markdown('<div class="frames-label">Sampled frames (16 evenly spaced)</div>', unsafe_allow_html=True)
                        row1 = st.columns(8)
                        row2 = st.columns(8)
                        for i, frm in enumerate(frames[:8]):
                            row1[i].image(frm, use_container_width=True)
                        for i, frm in enumerate(frames[8:]):
                            row2[i].image(frm, use_container_width=True)

                        video_tensor = frames_to_tensor(frames).to(device)
                        nc = len(VIDEO_CLASSES)

                        if not is_trimodal:
                            model = CNN_LSTM(num_classes=nc)
                            model = load_checkpoint(model, vid_ckpt)
                            with torch.no_grad():
                                probs = torch.softmax(model(video_tensor), dim=1)[0].numpy()

                        else:
                            # ── Build text + audio features ──────────────────
                            ACTION_TEMPLATES = [
                                'this is {adj} footage of {cls} {action}',
                                'the video is showing {adj} {cls} {action} in {place}',
                                'captured footage featuring {cls} {action} with {adj} conditions',
                            ]
                            ACTION_WORDS = {
                                'Basketball':   {'adj':'competitive','action':'players dribbling','place':'indoor court'},
                                'Biking':       {'adj':'outdoor',    'action':'rider cycling on road','place':'open track'},
                                'Bowling':      {'adj':'indoor',     'action':'player throwing ball','place':'bowling alley'},
                                'CliffDiving':  {'adj':'extreme',    'action':'athlete jumping from cliff','place':'ocean shore'},
                                'GolfSwing':    {'adj':'professional','action':'golfer swinging club','place':'golf course'},
                                'HorseRiding':  {'adj':'outdoor',    'action':'rider on horseback','place':'open field'},
                                'Skiing':       {'adj':'winter',     'action':'skier going down slope','place':'snowy mountain'},
                                'Surfing':      {'adj':'coastal',    'action':'surfer riding wave','place':'ocean shore'},
                                'TennisSwing':  {'adj':'competitive','action':'player swinging racket','place':'tennis court'},
                                'SkateBoarding':{'adj':'urban',      'action':'skater performing tricks','place':'skate park'},
                            }

                            if vid_text_input and vid_text_input.strip():
                                caption = vid_text_input.strip()
                            else:
                                cls_pick = random.choice(VIDEO_CLASSES)
                                w = ACTION_WORDS[cls_pick]
                                tmpl = random.choice(ACTION_TEMPLATES)
                                caption = tmpl.format(cls=cls_pick.lower(), **w)
                                st.caption(f"Auto-generated caption: *\"{caption}\"*")

                            with st.spinner("Encoding text with DistilBERT…"):
                                txt_feat   = get_text_features(caption)        # (1, 768)

                            with st.spinner("Synthesising audio → MFCC…"):
                                audio_feat = get_audio_features(caption).unsqueeze(0)  # (1, 120)

                            model = TrimodalFusionModel(num_classes=nc)
                            model = load_checkpoint(model, vid_ckpt)
                            with torch.no_grad():
                                probs = torch.softmax(
                                    model(video_tensor, txt_feat, audio_feat), dim=1
                                )[0].numpy()

                        pred_class = VIDEO_CLASSES[int(np.argmax(probs))]
                        confidence = float(np.max(probs))

                        st.markdown(f"""
                        <div class="pred-badge">
                          <span>🎯</span>
                          <span class="pred-label">{pred_class}</span>
                          <span class="pred-conf">{confidence*100:.1f}% confidence</span>
                        </div>
                        """, unsafe_allow_html=True)

                        st.plotly_chart(
                            confidence_chart(probs, VIDEO_CLASSES),
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.info("Check that the checkpoint matches the selected architecture.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color:#3a4d6a; font-size:0.82rem; font-family:'Space Mono',monospace;">
  Deep Learning · News Content Recognition &nbsp;|&nbsp; Final Year Project · May 2026
</div>
""", unsafe_allow_html=True)