# Age Estimator 🧑‍💻

A real-time age estimation project built with **PyTorch** and **OpenCV**, with two ways to run it:

- a **desktop webcam window**, and
- a minimal **browser UI** backed by a **FastAPI** endpoint.

The model is a fine-tuned **ResNet-18** that regresses age directly from a face image. Faces are located with **MTCNN** ([facenet-pytorch](https://github.com/timesler/facenet-pytorch)), and the predicted age is drawn live over the video feed.

---

## ✨ Features

- **Transfer learning** — ImageNet-pretrained ResNet-18 with a custom regression head
- **Robust face detection** — MTCNN (`facenet-pytorch`) handles angles, lighting, and small faces, and filters low-confidence detections
- **Real-time inference** — Detects faces from a webcam feed and labels each face with its predicted age
- **Desktop + browser** — Native OpenCV window or a minimal browser UI (camera, start/stop button, status console)
- **REST API** — `POST /predict_face` accepts an image and returns face boxes, ages, and confidences
- **Lightweight dataset builder** — Parses UTKFace-style filenames into a CSV automatically
- **Sensible defaults** — MAE loss, AdamW, data augmentation, and best-checkpoint saving out of the box

---

## 📁 Project Structure

```
age_estimator/
├── app.py                      # FastAPI API + serves index.html
├── index.html                  # Minimal webcam UI (camera, start/stop, console)
├── best_age_model.pth          # Trained checkpoint (committed)
├── hf_space/                   # Standalone Gradio app for Hugging Face Spaces
│   ├── app.py
│   ├── README.md               # Space metadata (sdk, app_file, python_version)
│   └── requirements.txt        # CPU-only wheels
├── pyproject.toml              # Project metadata & dependencies (uv)
├── requirements.txt            # Pinned plain-pip equivalent
├── uv.lock                     # Lockfile
└── src/
    └── age_estimator/
        ├── __init__.py
        ├── data.py             # Scans images, builds dataset.csv
        ├── cnn_train.py        # Training + validation loop
        ├── main.py             # Real-time webcam age estimation (desktop)
        └── data/               # Dataset images (UTKFace format, git-ignored)
```

> `src/age_estimator/data/` and `dataset.csv` are **git-ignored**. The trained `best_age_model.pth` is **committed** so the deployed API has its weights; retrain and commit it again if you change the model.

Dataset files are expected to follow the **UTKFace** naming convention:

```
age_gender_race_date.jpg
e.g. 25_0_1_20170116174525125.jpg
```

`age` is the target; `gender` and `race` are parsed as extra metadata but are **not** used for training.

### Tech stack

| Layer | Tools |
| ----- | ----- |
| Model | PyTorch `2.14.1` + torchvision `0.29.1` (CUDA 13.0 wheels) |
| Face detection | MTCNN via `facenet-pytorch` `2.5.3` |
| API | FastAPI + Uvicorn |
| Hosted demo | Gradio on Hugging Face Spaces |
| Video / images | OpenCV, Pillow |
| Data | pandas, NumPy |
| Packaging | uv (`pyproject.toml` + `uv.lock`) |

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

### 6. Run the browser UI

The same model is also exposed as a **FastAPI** endpoint and used by `index.html`:

```bash
uv run uvicorn app:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/>.

The page is intentionally minimal — it contains only:

- a **webcam viewport**, with detected faces and ages drawn on an overlay canvas,
- a **Start / Stop Camera** button, and
- a **status console** for the API target, detections, and errors.

FastAPI serves `index.html` at `/`, so the UI and API share an origin. The script still targets `http://127.0.0.1:8000` explicitly (`API_HOST` / `API_PORT` near the top of the script), so the same page also works through VS Code Live Server (port `5500`) or when opened from disk (`file://`). CORS is enabled in `app.py`; change `API_PORT` if your API runs elsewhere.

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/` | `GET` | Serves the browser UI |
| `/predict_face` | `POST` | Accepts multipart form-data (`file` = image) and returns detected boxes + ages |
| `/docs` | `GET` | Auto-generated Swagger UI |

Response shape:

```json
{
  "faces_found": 1,
  "data": [{ "box": [x1, y1, x2, y2], "predicted_age": 27.4, "confidence": 0.998 }]
}
```

> The page captures a webcam frame every ~500 ms, POSTs it to `/predict_face`, filters detections below `0.50` confidence, and draws the returned boxes and age labels on the overlay. If the API is unreachable it backs off to one retry every 3 seconds instead of spamming requests.

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

1. Capture a frame — OpenCV in `main.py`, or a canvas snapshot sent to `app.py` from the browser.
2. Detect faces with **MTCNN** (`facenet_pytorch.MTCNN`), keeping detections above a confidence threshold.
3. Crop each face in RGB, add a 15% margin, resize to `224×224`, and normalize.
4. Run it through the loaded ResNet-18 and read the scalar age output.
5. Draw the bounding box and predicted age — on the OpenCV window or the HTML overlay canvas.

---

## ⚠️ Limitations

- MTCNN is heavier than Haar Cascade and can be slow on CPU or with many faces in frame.
- Age is treated as a pure regression target, so predictions are approximate (± a few years).
- The model is only as good as the training data — UTKFace has skewed age and demographic distributions.
- Accuracy drops on heavily blurred or very low-resolution webcam frames.
- The browser canvas API requires `localhost` or HTTPS, and the FastAPI server must be running for the web UI.

---

## 🛣️ Ideas for Improvement

- Try RetinaFace or MediaPipe for faster/more accurate detection than MTCNN.
- Add a gender/race multi-task head using the existing metadata.
- Experiment with deeper backbones (ResNet-50, EfficientNet) or ordinal age classification.
- Track and plot train/validation MAE curves.
- Add a CLI flag to accept image files in addition to webcam input.
- Add a `Dockerfile` and deploy the FastAPI app.

---

## ☁️ Deploying to Hugging Face Spaces (Gradio)

The `hf_space/` folder is a **self-contained Gradio Space**:

```
hf_space/
├── README.md          # Space config (sdk: gradio, app_file: app.py, python_version: 3.12)
├── app.py             # Webcam upload -> MTCNN -> ResNet-18 -> annotated image + JSON
└── requirements.txt   # torch 2.13.0 + torchvision 0.28.0 (ZeroGPU & CPU compatible)
```

> `requirements.txt` deliberately pins torch **2.13.0** / torchvision **0.28.0**: these are on the ZeroGPU allow-list *and* run on CPU basic, so the Space builds whichever hardware is selected. `app.py` runs inference on CPU (set `DEVICE=cuda` only on real GPU hardware).

### Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Choose **Gradio** as the SDK, **CPU basic (free)** as the hardware, and create the Space.
3. Upload the three files from `hf_space/` (via **Files → Add file → Upload files**), or push with Git:

   ```bash
   cd hf_space
   git init
   git add .
   git commit -m "Age estimator Gradio Space"
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push -u space main
   ```

4. The Space builds automatically. `README.md` inside `hf_space/` already sets `app_file: app.py`, `sdk: gradio`, and `python_version: "3.12"`.

### Model weights

The Space does **not** bundle the 43 MB checkpoint. On startup, `hf_space/app.py` looks for `best_age_model.pth` next to itself, then in `/tmp`, and finally downloads it from the main GitHub repo. To avoid that download (faster cold starts), upload `best_age_model.pth` from the repo root into the Space root.

To use different weights, set a `MODEL_URL` environment variable in the Space settings.

> Free Spaces sleep after inactivity, so the first request after a pause may take a while (cold start + model load). CPU inference is a few seconds per image.

---

## 👤 Author

**JellyFishJuggler** — [guptasrijan85@gmail.com](mailto:guptasrijan85@gmail.com)
