import torch
from sklearn.metrics import classification_report
from models.news_cnn import NewsCNN
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

checkpoint = torch.load(config.MODEL_PATH)
classes = checkpoint['class_names']

model = NewsCNN(num_classes=len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=classes))
