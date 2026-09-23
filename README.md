# FDM Defect Segmentation

A small FastAPI web app for inspecting FDM 3D print photos. The frontend is plain HTML, CSS, and JavaScript. It uses a frozen YOLO26 instance segmentation model to identify Cracking, Layer_Shifting, Stringing, and Warping. In the UI, Layer_Shifting appears as “Layer Shifting.”

The checkpoint must stay at `model/selected_highest_mask_mAP50.pt`. Its SHA256 is `33e2edae67c37d99c0138383d2a05db338c3f2857fd91a1c75f16f9a9b6f7ede`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Upload a JPG or PNG image, then select **Analyze image**. The app shows the original image, segmentation overlay, detection count, and each instance's class and confidence. Uploads are limited to 20 MB and 25 megapixels.

## Inference

`POST /api/analyze` accepts a multipart image under the `file` field and returns JSON with `segmented_image` (a PNG data URL), `total_detections`, and `detections`. Inference uses `imgsz=960`, `conf=0.25`, `iou=0.7`, and `verbose=False`. Masks, boxes, labels, and confidence are rendered with `result.plot()`.

## Deployment

Run with `python3 -m uvicorn app:app --host 0.0.0.0 --port 8000`, or set the port required by your host. The host needs enough RAM for the Ultralytics model and inference. Serve the app as a single process unless you have sized the host for a model copy per worker.

The frontend is in `static/`. The heading font is the user-provided `PP Mondwest` file in `static/fonts/`.

The drag-and-drop overlay uses the Picture, Clip 1, and Share 3 icons from [Streamline Pixel](https://www.streamlinehq.com/icons/pixel), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Local copies are in `static/icons/`.
