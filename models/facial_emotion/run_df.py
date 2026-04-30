import cv2
import csv
from datetime import datetime
from deepface import DeepFace
import numpy as np

# Configuration
CSV_FILE = 'emotion_detections_deepface.csv'

# Initialize CSV file
with open(CSV_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Emotion', 'Confidence', 'All_Emotions'])

print(f"CSV file '{CSV_FILE}' initialized.")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("\nStarting emotion detection with DeepFace...")
print("Press 'q' to quit")
print(f"Saving detections to: {CSV_FILE}")

frame_count = 0
save_interval = 30  # Save to CSV every 30 frames (~1 second at 30fps)
skip_frames = 5  # Process every 5th frame for better performance

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Process only every nth frame for performance
        if frame_count % skip_frames == 0:
            try:
                # Analyze emotions using DeepFace
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )

                # Handle both single face and multiple faces
                if isinstance(result, list):
                    faces_data = result
                else:
                    faces_data = [result]

                for face_data in faces_data:
                    # Get face region
                    region = face_data['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    # Get dominant emotion
                    emotion = face_data['dominant_emotion']
                    
                    # Get all emotion scores
                    emotions_dict = face_data['emotion']
                    confidence = emotions_dict[emotion]

                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Display emotion and confidence
                    label = f"{emotion}: {confidence:.1f}%"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Display top 3 emotions
                    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    y_offset = y + h + 20
                    for emo, score in sorted_emotions:
                        text = f"{emo}: {score:.1f}%"
                        cv2.putText(frame, text, (x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20

                    # Save to CSV periodically
                    if frame_count % save_interval == 0:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        all_emotions_str = ', '.join([f"{k}: {v:.2f}" for k, v in emotions_dict.items()])
                        
                        with open(CSV_FILE, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, emotion, f"{confidence:.2f}", all_emotions_str])
                        
                        print(f"Saved: {timestamp} - {emotion} ({confidence:.1f}%)")

            except Exception as e:
                # If no face detected or error, just continue
                pass

        # Display frame
        cv2.imshow('DeepFace Emotion Detection', frame)

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