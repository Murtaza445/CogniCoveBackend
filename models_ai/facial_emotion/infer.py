# models_ai/facial_emotion/infer.py

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# Emotion labels (same as training)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ================= MODEL ARCHITECTURE =================
class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FER_CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ================= INFERENCE CLASS =================
class FacialEmotionModel:
    def __init__(self, model_path="models/facial_emotion/best_fer_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = FER_CNN(num_classes=len(EMOTIONS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predict(self, frame):
        """
        Input:
            frame (numpy array, BGR image from webcam/frontend)
        Output:
            dict: {emotion, confidence}
        """

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )

            if len(faces) == 0:
                return {"emotion": "no_face", "confidence": 0.0}

            # Take first face only
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            face_tensor = self.transform(face_roi).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probs, 1)

                emotion = EMOTIONS[predicted.item()]
                confidence_score = confidence.item()

            return {
                "emotion": emotion,
                "confidence": round(confidence_score, 4)
            }

        except Exception as e:
            print("Facial Emotion Error:", e)
            return {"emotion": "error", "confidence": 0.0}