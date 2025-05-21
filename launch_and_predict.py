import cv2
import time
import torch
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from speaker import speak

# Настройка устройства (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение трансформаций для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка модели ResNet18 и замена финального слоя
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # Два класса: good и bad bottles
model.load_state_dict(torch.load('bottle_classifier.pth', map_location=device))
model.to(device)
model.eval()


# Функция для предсказания изображения
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities[0][1].item() * 100, probabilities[0][0].item() * 100


# Инициализация камеры
cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # Захватываем первый кадр для начальной инициализации
previous_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_image is not None:
            # Вычисляем SSIM между текущим и предыдущим изображением
            score, _ = compare_ssim(previous_image, current_image, full=True)
            print("SSIM: ", score)

            if score < 0.7:  # Порог чувствительности к изменениям
                print("Significant change detected.")

                # Преобразование захваченного кадра для предсказания
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                good_prob, bad_prob = predict_image(pil_image)

                # Выводим вероятности для каждого класса
                print(f'Good Bottle Probability: {good_prob:.2f}%')
                print(f'Bad Bottle Probability: {bad_prob:.2f}%')
                if good_prob > 60:
                    speak("good.mp3")
                elif good_prob < 50:
                    speak("bad.mp3")

        previous_image = current_image  # Обновляем предыдущий кадр
        time.sleep(1)  # Пауза на 1 секунду

finally:
    cap.release()
    cv2.destroyAllWindows()
