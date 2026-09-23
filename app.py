"""Web UI and inference API for FDM defect segmentation."""

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "selected_highest_mask_mAP50.pt"
SUPPORTED_CLASSES = {"Cracking", "Layer_Shifting", "Stringing", "Warping"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

app = FastAPI(title="FDM Defect Segmentation")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
model_lock = Lock()


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))
    if model.task != "segment":
        raise ValueError("Checkpoint is not an instance segmentation model")

    names = model.names
    class_names = set(names.values() if isinstance(names, dict) else names)
    if len(names) != 4 or class_names != SUPPORTED_CLASSES:
        raise ValueError("Checkpoint does not contain the four supported classes")

    return model


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/analyze")
def analyze_image(file: UploadFile = File(...)) -> dict:
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 20 MB limit.")

    try:
        with open_image(BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, "Upload a JPG or PNG image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image resolution is too large.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "This file could not be opened as an image.") from None

    try:
        model = load_model()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, "The segmentation model is unavailable.") from exc

    try:
        with model_lock:
            result = model.predict(
                source=image,
                imgsz=960,
                conf=0.25,
                iou=0.7,
                verbose=False,
            )[0]

        plotted = Image.fromarray(result.plot()[..., ::-1])
        output = BytesIO()
        plotted.save(output, format="PNG")
        boxes = result.boxes
        detections = (
            [
                {
                    "class_name": result.names[int(class_id)].replace("Layer_Shifting", "Layer Shifting"),
                    "confidence": round(float(confidence), 4),
                }
                for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist())
            ]
            if boxes is not None
            else []
        )
    except Exception as exc:
        raise HTTPException(500, "Image analysis failed. Please try another image.") from exc

    return {
        "segmented_image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
        "total_detections": len(detections),
        "detections": detections,
    }
