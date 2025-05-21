import cv2
import time
import torch
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from speaker import speak

# Set up device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load ResNet18 model and replace final layer
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: good and bad bottles
model.load_state_dict(torch.load('bottle_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Function to predict image class probabilities
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities[0][1].item() * 100, probabilities[0][0].item() * 100

# Initialize camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # Capture first frame for initial setup
previous_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_image is not None:
            # Calculate SSIM between current and previous frame
            score, _ = compare_ssim(previous_image, current_image, full=True)
            print("SSIM: ", score)

            if score < 0.8:  # Change sensitivity threshold
                print("Significant change detected.")

                # Convert captured frame for prediction
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                good_prob, bad_prob = predict_image(pil_image)

                # Output the probabilities for each class
                print(f'Good Bottle Probability: {good_prob:.2f}%')
                print(f'Bad Bottle Probability: {bad_prob:.2f}%')

                # Speak result based on probability
                if good_prob > 60:
                    speak("goodITAL.mp3")
                elif good_prob < 50:
                    speak("bad.mp3")

        previous_image = current_image  # Update previous frame
        time.sleep(1)

finally:
    cap.release()
    cv2.destroyAllWindows()
