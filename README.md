# 📰 News Content Recognition System (Image & Video)

A Deep Learning based system to classify news media (Images & Videos) into categories such as **Politics** and **Sports**.

This project demonstrates:

- Custom CNN built from scratch
- Transfer Learning using ResNet18
- Image classification
- Video classification using frame-level prediction
- Performance comparison between models

---

## 📌 Project Motivation

With the rapid growth of digital media, news agencies handle large volumes of visual content daily.  
This system helps automatically categorize incoming news images and videos, reducing manual effort and improving efficiency.

---

## 🏗 Project Structure

```
news-project/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   ├── custom_cnn.py
│   └── resnet_model.py
│
├── train_custom.py
├── train_resnet.py
├── predict.py
├── video_predict.py
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd news-project
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision pillow opencv-python
```

---

## 📂 Dataset Format

Place dataset in the following structure:

```
dataset/
   train/
      politics/
      sport/
   test/
      politics/
      sport/
```

Each folder should contain corresponding images.

---

## 🏋️ Training

### 🔹 Train Custom CNN

```bash
python train_custom.py
```

Model will be saved as:

```
custom_model.pth
```

---

### 🔹 Train ResNet (Transfer Learning)

```bash
python train_resnet.py
```

Model will be saved as:

```
resnet_model.pth
```

---

## 🔍 Image Prediction

Place a test image in the project folder and update filename inside `predict.py`.

Run:

```bash
python predict.py
```

Example Output:

```
Prediction: sport
Confidence: 92.45%
Model Used: resnet
```

---

## 🎥 Video Prediction

Place video file in project folder.

Run:

```bash
python video_predict.py
```

The system:

- Extracts frames from the video
- Classifies each frame
- Uses majority voting
- Outputs final video category
