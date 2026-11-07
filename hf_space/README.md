---
title: Age Estimator
emoji: 🧑
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: Estimate age from a face photo with ResNet-18 + MTCNN
pinned: false
---

# Age Estimator

Real-time-style age estimation from a **webcam snapshot** or an **uploaded photo**.

- **Face detection:** MTCNN ([facenet-pytorch](https://github.com/timesler/facenet-pytorch))
- **Age regression:** fine-tuned ResNet-18 (`best_age_model.pth`)
- **Output:** annotated image with a box + predicted age per face, plus JSON with boxes and confidence scores

## How to use

1. Open the **Webcam** tab and take a photo, or switch to **Upload** and choose an image.
2. Press **Estimate Age** (the result also appears automatically after a webcam capture).
3. Check the annotated image and the JSON panel for exact values.

## Model weights

This folder does **not** include the 43 MB checkpoint. On startup the app downloads it from the main GitHub repository (`MODEL_URL`). To speed up cold starts, upload `best_age_model.pth` to the root of this Space — the app will use the local file instead.

If your weights live elsewhere, set the `MODEL_URL` environment variable in the Space settings.

## Notes

- Detections below `0.50` confidence are ignored.
- A face crop is padded by 15% to match the UTKFace-style training images.
- The free CPU tier is used; expect a few seconds per image.
