import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.news_cnn import NewsCNN
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(config.MODEL_PATH, map_location=device)
classes = checkpoint['class_names']

model = NewsCNN(num_classes=len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

st.title("📰 News Content Recognition System")

uploaded_file = st.file_uploader("Upload News Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")


    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    st.write("### Prediction:", classes[predicted.item()])
    st.write("### Confidence:", round(confidence.item()*100,2), "%")
