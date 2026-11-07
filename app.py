import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

app = FastAPI(title="Age Estimator API")

# The HTML page may be opened from file:// or another origin, so allow
# cross-origin requests during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)

model_path = BASE_DIR / "best_age_model.pth"

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.get("/")
def home():
    """Serve the dashboard so the API and page share the same origin."""
    return FileResponse(BASE_DIR / "index.html")


@app.post('/predict_face')
async def predict_face(file: UploadFile = File(...)):
    """
    Detect every face in the uploaded image and estimate its age.

    Input:  multipart/form-data with a single image field named `file`.
    Output: { faces_found: int, data: [{ box, predicted_age, confidence }] }
    """
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    boxes, probs = mtcnn.detect(img)
    results = []

    if boxes is not None:
        img_w, img_h = img.size

        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(b) for b in box]

            # MTCNN gives a tight box; UTKFace training images have a margin,
            # so pad by 15% and then clamp inside the image.
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            face_crop = img.crop((x1, y1, x2, y2))

            input_tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor).item()

            results.append({
                "box": [x1, y1, x2, y2],
                "predicted_age": round(prediction, 1),
                "confidence": round(float(prob), 3) if prob is not None else None,
            })

    return {"faces_found": len(results), "data": results}
