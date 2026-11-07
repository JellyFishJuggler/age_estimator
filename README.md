# Age Estimator 🧑‍💻

A real-time age estimation app built with **PyTorch** and **OpenCV**. It fine-tunes a pretrained **ResNet-18** to regress age directly from a face image, then uses a webcam feed with Haar Cascade face detection to predict and display age live on screen.

---

## ✨ Features

- **Transfer learning** — ImageNet-pretrained ResNet-18 with a custom regression head
- **Real-time inference** — Detects faces from a webcam feed and labels each face with its predicted age
- **Lightweight dataset builder** — Parses UTKFace-style filenames into a CSV automatically
- **Sensible defaults** — MAE loss, AdamW, data augmentation, and best-checkpoint saving out of the box

---

## 📁 Project Structure

```
age_estimator/
├── best_age_model.pth          # Saved model checkpoint (created after training)
├── pyproject.toml              # Project metadata & dependencies (uv)
├── requirements.txt
└── src/
    └── age_estimator/
        ├── __init__.py
        ├── data.py             # Scans images, builds dataset.csv
        ├── cnn_train.py        # Training + validation loop
        ├── main.py             # Real-time webcam age estimation
        └── data/               # Dataset images (UTKFace format)
```

Dataset files are expected to follow the **UTKFace** naming convention:

```
age_gender_race_date.jpg
e.g. 25_0_1_20170116174525125.jpg
```

`age` is the target; `gender` and `race` are parsed as extra metadata but are **not** used for training.

---

## 🚀 Getting Started

### 1. Install dependencies

Requires **Python ≥ 3.14** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

> `torch` and `torchvision` are pulled from the `pytorch-cu130` index (CUDA 13.0) as configured in `pyproject.toml`. If you don't have a compatible GPU, install CPU builds of PyTorch manually.

<details>
<summary>Using pip instead</summary>

```bash
pip install -r requirements.txt
```

`requirements.txt` is pinned to the exact versions in the project's `.venv`. To also install the package itself (and the `age-estimator` command), run `pip install -e .` afterwards.

</details>

### 2. Add the dataset

Place the face images inside:

```
src/age_estimator/data/
```

### 3. Build the dataset CSV

```bash
uv run python -m age_estimator.data
```

This scans every `.jpg` in `src/age_estimator/data/`, extracts `age`, `gender`, and `race` from the filename, and writes `dataset.csv`. Corrupted or oddly named files are skipped.

> The CSV stores **absolute file paths**, so re-run this step whenever you move the project to a different location.

### 4. Train the model

```bash
uv run python src/age_estimator/cnn_train.py
```

The training script automatically uses CUDA when available and saves the best checkpoint (lowest validation MAE) to `best_age_model.pth`. Expect per-epoch output like:

```
Epoch [1/15] | Train MAE: 8.1234 years | Val MAE: 7.5678 years
--> Saved Best Model Checkpoint
```

### 5. Run real-time inference

```bash
uv run python src/age_estimator/main.py
```

A window titled **"Custom PyTorch Real-Time Age Estimator"** opens with your webcam feed. Each detected face gets a bounding box and an `Age: xx.x yrs` label.

Press **`q`** to quit.

---

## 🧠 How It Works

**Training pipeline**

| Component   | Details                                                                 |
| ----------- | ----------------------------------------------------------------------- |
| Backbone    | ResNet-18 (ImageNet weights)                                            |
| Head        | `Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1)`             |
| Loss        | `L1Loss` (MAE in years)                                                 |
| Optimizer   | AdamW, `lr=1e-4`, `weight_decay=1e-3`                                   |
| Batch size  | 32                                                                      |
| Epochs      | 15                                                                      |
| Split       | 90% train / 10% validation                                              |
| Augmentation| Resize `224×224`, random horizontal flip, color jitter (0.1/0.1)        |
| Normalization | ImageNet stats — mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

**Inference pipeline**

1. Capture a frame from the webcam with OpenCV.
2. Detect faces using `haarcascade_frontalface_default.xml`.
3. Convert each face crop to RGB, resize to `224×224`, and normalize.
4. Run it through the loaded ResNet-18 and read the scalar age output.
5. Draw the bounding box and predicted age on the frame.

---

## ⚠️ Limitations

- Haar Cascade detection can miss faces in poor lighting, at odd angles, or when partially occluded.
- Age is treated as a pure regression target, so predictions are approximate (± a few years).
- The model is only as good as the training data — UTKFace has skewed age and demographic distributions.
- Accuracy drops on heavily blurred or very low-resolution webcam frames.

---

## 🛣️ Ideas for Improvement

- Replace Haar Cascade with a modern detector (MTCNN, RetinaFace, MediaPipe).
- Add a gender/race multi-task head using the existing metadata.
- Experiment with deeper backbones (ResNet-50, EfficientNet) or ordinal age classification.
- Track and plot train/validation MAE curves.
- Add a CLI flag to accept image files in addition to webcam input.

---

## 👤 Author

**JellyFishJuggler** — [guptasrijan85@gmail.com](mailto:guptasrijan85@gmail.com)
