import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet_model import get_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = get_resnet_model(len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

epochs = 10

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Accuracy: {100*correct/total:.2f}%")

torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': train_dataset.classes,
    'model_type': 'resnet'
}, "resnet_model.pth")
