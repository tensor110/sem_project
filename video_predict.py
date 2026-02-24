import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import Counter

from models.resnet_model import get_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load("resnet_model.pth", map_location=device)
classes = checkpoint['class_names']

model = get_resnet_model(len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

video_path = "test_video1.mp4"

cap = cv2.VideoCapture(video_path)

predictions = []

frame_count = 0
frame_skip = 10   # process 1 frame every 30 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(probs, 1)

        predictions.append(classes[pred.item()])

    frame_count += 1

cap.release()

# Majority voting
final_prediction = Counter(predictions).most_common(1)[0][0]

print("Frame Predictions:", predictions)
print("Final Video Prediction:", final_prediction)
