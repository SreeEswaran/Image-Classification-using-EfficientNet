import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Data path
test_dir = 'data/test'

# Data transformations
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(test_dataset.classes))
model.load_state_dict(torch.load('models/efficientnet_model.pth'))

# Evaluate the model
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
