"""
Hugging Face Spaces entry point (Gradio SDK).

Webcam snapshot or uploaded photo -> MTCNN face detection -> fine-tuned
ResNet-18 age regression -> annotated image + JSON results.

The trained checkpoint is not bundled with this folder. If
``best_age_model.pth`` is absent, it is downloaded from ``MODEL_URL``
(default: the main GitHub repository) on startup.
"""

import os
import tempfile
import urllib.request
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "best_age_model.pth"
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://raw.githubusercontent.com/JellyFishJuggler/age_estimator/main/best_age_model.pth",
)

# Prefer a checkpoint uploaded next to this file, then one in the parent
# directory (when this folder lives inside the main repo), then a cached copy
# in /tmp, and finally download it from MODEL_URL.
def _resolve_model_path() -> Path:
    candidates = [
        BASE_DIR / MODEL_FILENAME,
        BASE_DIR.parent / MODEL_FILENAME,
        Path(tempfile.gettempdir()) / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    print(f"Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, candidates[-1])
    return candidates[-1]


model_path = _resolve_model_path()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MIN_CONFIDENCE = 0.50


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def predict(image: Image.Image):
    """Detect faces, estimate ages, and return an annotated image."""
    if image is None:
        raise gr.Error("Please capture a webcam photo or upload an image first.")

    img = image.convert("RGB")
    img_w, img_h = img.size

    boxes, probs = mtcnn.detect(img)
    draw = ImageDraw.Draw(img)
    font_size = max(16, img_w // 32)
    font = _font(font_size)
    line_width = max(2, img_w // 400)

    results = []

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is None or prob < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Match the training crops: add a 15% margin, then clamp.
            pad = int(max(x2 - x1, y2 - y1) * 0.15)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            face = img.crop((x1, y1, x2, y2))
            tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                age = model(tensor).item()

            # Box
            draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=line_width)

            # Label with a dark background for readability
            label = f"{age:.1f} yrs"
            text_x, text_y = x1 + 6, max(2, y1 - int(font_size * 1.6))
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(
                [
                    text_bbox[0] - 4,
                    text_bbox[1] - 2,
                    text_bbox[2] + 4,
                    text_bbox[3] + 2,
                ],
                fill=(2, 6, 23),
            )
            draw.text((text_x, text_y), label, fill=(74, 222, 128), font=font)

            results.append({
                "box": [x1, y1, x2, y2],
                "predicted_age": round(age, 1),
                "confidence": round(float(prob), 3),
            })

    if results:
        ages = ", ".join(f"{row['predicted_age']} yrs" for row in results)
        status = f"Detected {len(results)} face(s): {ages}"
    else:
        status = "No face detected — try better lighting or a clearer photo."

    return img, results, status


with gr.Blocks(title="Age Estimator") as demo:
    gr.Markdown(
        """
        # Age Estimator
        Face detection with **MTCNN** + age regression with a fine-tuned **ResNet-18**.
        Capture a photo with your webcam or upload an image, then press **Estimate Age**.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                sources=["webcam", "upload"],
                label="Webcam / Upload",
            )
            estimate_btn = gr.Button("Estimate Age", variant="primary")

        with gr.Column():
            image_output = gr.Image(type="pil", label="Detections")
            json_output = gr.JSON(label="Faces")
            status_output = gr.Textbox(label="Status", interactive=False)

    estimate_btn.click(
        predict,
        inputs=image_input,
        outputs=[image_output, json_output, status_output],
    )


if __name__ == "__main__":
    demo.launch()
