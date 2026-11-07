import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN


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


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # -----------------------------------------------------------------------
    # OLD: Haar Cascade detection (grayscale image, boxes only).
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #
    # NEW: MTCNN detection (RGB image, boxes + confidence scores).
    # `mtcnn.detect()` returns (boxes, probs):
    #   boxes -> array of (x1, y1, x2, y2) floats in the original frame
    #   probs -> confidence per face, or (None, None) when no face is found
    # -----------------------------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb)

    # OLD: for (x, y, w, h) in faces:
    # NEW: iterate over MTCNN boxes only when at least one face was found
    if boxes is not None:
        frame_h, frame_w = frame.shape[:2]

        # OLD: x1, y1, x2, y2 = x, y, x + w, y + h
        # NEW: unpack each MTCNN box and keep its confidence score
        for box, prob in zip(boxes, probs):
            # Haar Cascade had no confidence score to filter on.
            # MTCNN does, so skip low-confidence detections.
            if prob is None or prob < 0.90:
                continue

            # OLD: x1, y1, x2, y2 = x, y, x + w, y + h
            # NEW: convert float box coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # padding yahan
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            x1, y1 = x1 - pad_x, y1 - pad_y
            x2, y2 = x2 + pad_x, y2 + pad_y

            # clamping yahan (pehle se maujood hai)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            # NEW: skip degenerate boxes after clamping.
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            # OLD: face_roi = frame[y1:y2, x1:x2]
            #      if face_roi.size != 0:
            # NEW: crop directly from the RGB frame that MTCNN already used.
            face_rgb = rgb[y1:y2, x1:x2]

            # OLD: face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            # NEW: already RGB — no conversion needed here.
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