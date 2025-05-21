# work.py
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the same model architecture and replace final layer
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: good and bad bottles

# Load trained model weights
model.load_state_dict(torch.load('bottle_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Function to predict class probabilities for an image
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # Print probabilities for each class
    print(f'Good Bottle Probability: {probabilities[0][1].item() * 100:.2f}%')
    print(f'Bad Bottle Probability: {probabilities[0][0].item() * 100:.2f}%')

# Run prediction on a sample image
if __name__ == '__main__':
    image_path = 'bad_59.jpg'  # Specify the image path
    predict_image(image_path)
