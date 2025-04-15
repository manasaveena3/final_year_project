import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT & ResNet
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define dataset paths
train_dir = "train"
val_dir = "val"
test_dir = "test"


# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import torchvision.models as models
from transformers import ViTModel

# Load Pretrained ResNet
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer

# Load Pretrained Vision Transformer (ViT)
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Freeze ResNet and ViT layers
for param in resnet.parameters():
    param.requires_grad = False

for param in vit.parameters():
    param.requires_grad = False

import torch.nn as nn

class HybridViTResNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTResNet, self).__init__()
        
        # ResNet Feature Extractor
        self.resnet = resnet
        self.resnet_fc = nn.Linear(2048, 512)  # ResNet output to 512
        
        # Vision Transformer Feature Extractor
        self.vit = vit
        self.vit_fc = nn.Linear(768, 512)  # ViT output to 512

        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 256),  # Merge ResNet & ViT features
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, x):
        resnet_features = self.resnet(x).view(x.size(0), -1)
        resnet_features = self.resnet_fc(resnet_features)

        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        vit_features = self.vit_fc(vit_features)

        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        output = self.fc(combined_features)

        return output
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridViTResNet(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 40

# Lists to store metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
print('started trainer')
# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0  # Track training accuracy
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy per step
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Loss: {train_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "hybrid_vit_resnet3.pth")
print("Model saved successfully.")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

import torch.optim as optim

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridViTResNet(num_classes=len(train_dataset.classes)).to(device)

import torch.optim as optim

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridViTResNet(num_classes=len(train_dataset.classes)).to(device)

model.load_state_dict(torch.load("hybrid_vit_resnet.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS =10

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0  # Track training accuracy
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy per step
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "hybrid_vit_resnet2.pth")
print("Model saved successfully.")


# Load the trained model
model.load_state_dict(torch.load("hybrid_vit_resnet2.pth"))
model.eval()

# Test the model
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"📊 Test Accuracy: {100 * correct / total:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix
import torch

# Load the trained model
model.load_state_dict(torch.load("hybrid_vit_resnet2.pth"))
model.eval()

# Initialize lists for predictions and ground truth labels
all_preds = []
all_labels = []

# Test the model
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store predictions and true labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute classification metrics
print(f"📊 Test Accuracy: {100 * correct / total:.2f}%")

# Generate classification report
print("\n🔹 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Generate confusion matrix
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
