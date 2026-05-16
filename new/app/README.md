# Deep Learning News Content Recognition — Streamlit App

## 📁 Folder Structure
```
app/
├── app.py                        ← Entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml               ← Dark theme + upload size
└── pages/
    ├── 01_overview.py            ← Project overview
    ├── 02_image_models.py        ← ResNet50 / VGG16 / Custom CNN results
    ├── 03_image_fusion.py        ← Splice + Weighted fusion results
    ├── 04_video_model.py         ← CNN+LSTM results (actual training numbers)
    ├── 05_video_fusion.py        ← Trimodal fusion results
    └── 06_live_demo.py           ← Upload image/video → live prediction
```

---

## 🚀 Deploy to Streamlit Cloud (Step-by-Step)

### 1. Create a GitHub repository
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/news-content-recognition.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set **Main file path** → `app.py`
6. Click **Deploy** ✅

Your app will be live at:
`https://YOUR_USERNAME-news-content-recognition-app-XXXX.streamlit.app`

---

## 📊 Update Your Actual Results

Before presenting to the panel, update the classification report numbers in:

- `pages/02_image_models.py`  → `REPORT` dict
- `pages/03_image_fusion.py`  → `FUSION_REPORT` dict
- `pages/04_video_model.py`   → `REPORT` dict (already has your real numbers from training)
- `pages/05_video_fusion.py`  → `TRIMODAL_REPORT` dict

---

## 📸 Upload Your PNG Files During the Demo

In sections **02, 03, 04, 05** there are file uploaders for your saved PNGs.
Download from Drive and upload them live during the panel presentation.

Files to prepare:
```
From Drive → ContentRecognition/results/image/
  ✅ model_comparison.png
  ✅ training_curves.png
  ✅ confusion_matrices.png
  ✅ fusion_curves.png
  ✅ fusion_confusion_matrices.png

From Drive → ContentRecognition/results/video/
  ✅ cnn_lstm_results.png
  ✅ trimodal_results.png
```

---

## 🔮 Live Demo — What to Bring

For the Live Demo tab you need:
1. **Model checkpoints (.pth files)** downloaded from Google Drive
2. **A few test images** (scene photos — buildings, forest, glacier, etc.)
3. **A short video clip** (MP4, ~5-10 seconds of a sports action)

The app runs inference on CPU, so predictions take 2-5 seconds.

---

## ⚠️ Important Notes

- Streamlit Cloud is **CPU only** — inference uses `torch.device("cpu")`
- Max upload size is set to **500MB** in config.toml
- `num_workers=0` in DataLoader (Streamlit doesn't support forked processes)
- `torch.load(..., map_location='cpu')` handles GPU-trained checkpoints correctly

---

## 🖥️ Run Locally (for testing)

```bash
pip install -r requirements.txt
streamlit run app.py
```
