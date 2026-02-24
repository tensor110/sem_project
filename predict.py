import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.news_cnn import NewsCNN
from models.resnet_model import get_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = "custom_model.pth"
MODEL_PATH = "resnet_model.pth"

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['class_names']
model_type = checkpoint['model_type']

if model_type == 'custom':
    model = NewsCNN(len(classes))
else:
    model = get_resnet_model(len(classes))

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

image = Image.open("/Users/as-mac-1248/Documents/Learning/Sem Project/dataset/test/politics/test_image_politics3.png").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probs = F.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)

print("Prediction:", classes[pred.item()])
print("Confidence:", round(conf.item()*100,2), "%")
print("Model Used:", model_type)
