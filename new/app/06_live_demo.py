import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import tempfile
import cv2
import plotly.graph_objects as go

st.title("🔮 Live Demo")
st.markdown("Upload an image or video and get a real-time prediction from the trained models.")
st.markdown("---")

# ── Constants ─────────────────────────────────────────────────────
IMAGE_CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
VIDEO_CLASSES = [
    'Basketball', 'Biking', 'Bowling', 'CliffDiving',
    'GolfSwing', 'HorseRiding', 'Skiing', 'Surfing',
    'TennisSwing', 'SkateBoarding'
]
FRAMES_PER_VIDEO = 16

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cpu")   # Streamlit Cloud has no GPU

# ── Model builders ────────────────────────────────────────────────
def build_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return m

def build_vgg16(num_classes):
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
            nn.Linear(50176, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.classifier(x)

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512, num_layers=2):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])

# ── Checkpoint loader (cached) ────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    """
    Load model from uploaded checkpoint.
    Returns None if checkpoint not available.
    """
    return None   # Populated dynamically when user uploads checkpoint

def predict_image(model, img_pil):
    model.eval()
    tensor = IMG_TRANSFORM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0].numpy()
    return probs

def extract_frames_from_video(video_bytes, num_frames=16):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap    = cv2.VideoCapture(tmp_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); os.unlink(tmp_path); return None

    indices   = set(np.linspace(0, total - 1, num_frames, dtype=int))
    frames    = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx in indices:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)
    return frames if len(frames) == num_frames else None

def frames_to_tensor(frames):
    tensors = [IMG_TRANSFORM(Image.fromarray(f)) for f in frames]
    return torch.stack(tensors).unsqueeze(0)   # (1, 16, 3, 224, 224)

def confidence_chart(probs, class_names, title):
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=class_names,
        orientation="h",
        marker_color=["#2B5BA8" if p == max(probs) else "#AEB6BF" for p in probs],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 115], title="Confidence (%)"),
        height=max(300, len(class_names) * 35),
        margin=dict(t=40, l=140, r=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖼️ Image Prediction", "🎬 Video Prediction", "📊 Model Comparison"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Image Scene Classification")
    st.markdown("Classes: buildings · forest · glacier · mountain · sea · street")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Step 1 — Upload your model checkpoint (.pth)**")
        ckpt_file = st.file_uploader(
            "Model checkpoint", type=["pth"], key="img_ckpt"
        )
        model_choice = st.selectbox(
            "Which model is this checkpoint?",
            ["ResNet50", "VGG16", "Custom CNN"],
            key="img_model_choice"
        )

        st.markdown("**Step 2 — Upload an image to classify**")
        img_file = st.file_uploader(
            "Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="img_upload"
        )

    with col_right:
        if img_file:
            img_pil = Image.open(img_file).convert("RGB")
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    if img_file and ckpt_file:
        with st.spinner("Loading model and running inference..."):
            try:
                num_classes = len(IMAGE_CLASSES)
                if model_choice == "ResNet50":
                    model = build_resnet50(num_classes)
                elif model_choice == "VGG16":
                    model = build_vgg16(num_classes)
                else:
                    model = CustomCNN(num_classes)

                # Load weights from uploaded file
                ckpt_bytes = ckpt_file.read()
                with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                    f.write(ckpt_bytes); tmp_ckpt = f.name

                state = torch.load(tmp_ckpt, map_location=device)
                model.load_state_dict(state)
                os.unlink(tmp_ckpt)

                probs      = predict_image(model, img_pil)
                pred_class = IMAGE_CLASSES[np.argmax(probs)]
                confidence = np.max(probs)

                st.success(f"**Prediction: {pred_class.upper()}**  ({confidence*100:.1f}% confident)")
                st.plotly_chart(
                    confidence_chart(probs, IMAGE_CLASSES, "Class Confidence Scores"),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.info("Make sure the checkpoint matches the selected model architecture.")

    elif img_file and not ckpt_file:
        st.info("⬆️  Upload a model checkpoint to get a prediction.")
    elif ckpt_file and not img_file:
        st.info("⬆️  Upload an image to classify.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Video Action Recognition")
    st.markdown("Classes: Basketball · Biking · Bowling · CliffDiving · GolfSwing · HorseRiding · Skiing · Surfing · TennisSwing · SkateBoarding")

    col_left2, col_right2 = st.columns([1, 1])

    with col_left2:
        st.markdown("**Step 1 — Upload CNN+LSTM checkpoint (.pth)**")
        video_ckpt = st.file_uploader(
            "cnn_lstm_best.pth", type=["pth"], key="vid_ckpt"
        )
        st.markdown("**Step 2 — Upload a video clip (MP4 / AVI)**")
        video_file = st.file_uploader(
            "Video file", type=["mp4", "avi", "mov"], key="vid_upload"
        )

    with col_right2:
        if video_file:
            st.video(video_file)

    if video_file and video_ckpt:
        with st.spinner("Extracting frames and running inference..."):
            try:
                frames = extract_frames_from_video(video_file.read())
                if frames is None:
                    st.error("Could not extract 16 frames from the video. Try a longer clip.")
                else:
                    # Show sampled frames
                    st.markdown("**Sampled frames (16 evenly spaced):**")
                    frame_cols = st.columns(8)
                    for i, frame in enumerate(frames[:8]):
                        frame_cols[i].image(frame, use_container_width=True)
                    frame_cols2 = st.columns(8)
                    for i, frame in enumerate(frames[8:]):
                        frame_cols2[i].image(frame, use_container_width=True)

                    # Load model
                    model = CNN_LSTM(num_classes=len(VIDEO_CLASSES))
                    ckpt_bytes = video_ckpt.read()
                    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                        f.write(ckpt_bytes); tmp_ckpt = f.name
                    state = torch.load(tmp_ckpt, map_location=device)
                    model.load_state_dict(state)
                    model.eval()
                    os.unlink(tmp_ckpt)

                    video_tensor = frames_to_tensor(frames).to(device)
                    with torch.no_grad():
                        out   = model(video_tensor)
                        probs = torch.softmax(out, dim=1)[0].numpy()

                    pred_class = VIDEO_CLASSES[np.argmax(probs)]
                    confidence = np.max(probs)

                    st.success(f"**Prediction: {pred_class.upper()}**  ({confidence*100:.1f}% confident)")
                    st.plotly_chart(
                        confidence_chart(probs, VIDEO_CLASSES, "Action Class Confidence"),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Inference failed: {e}")

    elif video_file and not video_ckpt:
        st.info("⬆️  Upload the CNN+LSTM checkpoint to get a prediction.")
    elif video_ckpt and not video_file:
        st.info("⬆️  Upload a video clip to classify.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 All Models — Side-by-side Comparison")

    st.markdown("#### Image Models")
    img_data = {
        "Model"           : ["ResNet50", "VGG16", "Custom CNN", "Splice Fusion", "Weighted Fusion"],
        "Modality"        : ["Image", "Image", "Image", "Image+Text", "Image+Text"],
        "Val Accuracy"    : ["92.4%", "92.1%", "88.5%", "94.5%", "93.8%"],
        "Macro F1"        : ["0.92", "0.92", "0.88", "0.94", "0.94"],
        "Pretrained"      : ["✅", "✅", "❌", "✅", "✅"],
        "Trainable Params": ["~262K", "~1.05M", "~26M", "~1.3M", "~1.1M"],
    }
    st.dataframe(img_data, use_container_width=True, hide_index=True)

    fig_img = go.Figure(go.Bar(
        x=img_data["Model"],
        y=[float(a.strip("%")) for a in img_data["Val Accuracy"]],
        marker_color=["#2B5BA8", "#27AE60", "#E74C3C", "#8E44AD", "#E67E22"],
        text=img_data["Val Accuracy"],
        textposition="outside",
        width=0.45,
    ))
    fig_img.update_layout(
        yaxis=dict(range=[84, 100], title="Accuracy (%)"),
        height=340, margin=dict(t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_img, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Video Models")
    vid_data = {
        "Model"           : ["CNN+LSTM", "Trimodal Fusion"],
        "Modality"        : ["Video", "Video+Text+Audio"],
        "Val Accuracy"    : ["97.37%", "98.1%"],
        "Macro F1"        : ["0.97", "0.98"],
        "Trainable Params": ["~7.48M", "~9.2M"],
    }
    st.dataframe(vid_data, use_container_width=True, hide_index=True)

    fig_vid = go.Figure(go.Bar(
        x=vid_data["Model"],
        y=[float(a.strip("%")) for a in vid_data["Val Accuracy"]],
        marker_color=["#27AE60", "#8E44AD"],
        text=vid_data["Val Accuracy"],
        textposition="outside",
        width=0.3,
    ))
    fig_vid.update_layout(
        yaxis=dict(range=[96, 100], title="Accuracy (%)"),
        height=300, margin=dict(t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vid, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Key Takeaways
    - **Pretrained models** (ResNet50, VGG16) outperform Custom CNN even with frozen backbones —
      ImageNet representations transfer well to scene classification.
    - **Multimodal fusion always outperforms unimodal** — adding text description improves image
      accuracy by ~2%, adding text+audio improves video accuracy by ~0.7%.
    - **Video modality is highly effective** — CNN+LSTM achieves 97.37% accuracy because
      action classes are visually very distinct (skiing vs. surfing vs. golf).
    - **Trimodal fusion** shows that even synthetic/generated text and audio carry
      useful complementary signal beyond the visual stream.
    """)
