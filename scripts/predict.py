import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import pandas as pd
import os

# Data transformations
predict_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)  # Adjust this line to match the number of classes
model.load_state_dict(torch.load('models/efficientnet_model.pth'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to predict image class
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Path to images for prediction
image_dir = 'data/test'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

# Predict classes for images
predictions = []
for image_path in image_paths:
    pred = predict_image(image_path, model, predict_transforms)
    predictions.append({'image': image_path, 'predicted_class': pred})

# Save predictions
pred_df = pd.DataFrame(predictions)
if not os.path.exists('results'):
    os.makedirs('results')
pred_df.to_csv('results/predictions.csv', index=False)
print(pred_df.head())
