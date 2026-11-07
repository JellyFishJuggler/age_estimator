import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)

model_path = "best_age_model.pth"

model.load_state_dict(torch.load(model_path, map_location=device))  
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size != 0:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_age = output.item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Age: {predicted_age:.1f} yrs'
            y_label = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Custom PyTorch Real-Time Age Estimator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()