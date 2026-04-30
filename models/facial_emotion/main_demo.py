import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import csv
from datetime import datetime
import os

# Define the same model architecture
class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FER_CNN, self).__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Fully Connected Layers
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


# Configuration
MODEL_PATH = 'best_fer_model.pth'  # Update this path
CSV_FILE = 'emotion_detections.csv'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Update based on your classes

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
print("Loading model...")
model = FER_CNN(num_classes=len(EMOTIONS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully!")

# Image preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize CSV file
with open(CSV_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Emotion', 'Confidence'])

print(f"CSV file '{CSV_FILE}' initialized.")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("\nStarting emotion detection...")
print("Press 'q' to quit")
print(f"Saving detections to: {CSV_FILE}")

frame_count = 0
save_interval = 30  # Save to CSV every 30 frames (~1 second at 30fps)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess face for model
            face_tensor = transform(face_roi).unsqueeze(0).to(device)

            # Predict emotion
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion = EMOTIONS[predicted.item()]
                confidence_score = confidence.item() * 100

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display emotion and confidence
            label = f"{emotion}: {confidence_score:.1f}%"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save to CSV periodically
            if frame_count % save_interval == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(CSV_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, emotion, f"{confidence_score:.2f}"])
                print(f"Saved: {timestamp} - {emotion} ({confidence_score:.1f}%)")

        # Display frame
        cv2.imshow('Emotion Detection', frame)

        frame_count += 1

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping...")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDetection completed. Results saved to '{CSV_FILE}'")
    print(f"Total frames processed: {frame_count}")