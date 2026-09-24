# FDM Defect Segmentation

A Streamlit app for inspecting FDM 3D print photos with the frozen YOLO26 instance segmentation checkpoint. It identifies Cracking, Layer_Shifting, Stringing, and Warping. The interface displays `Layer_Shifting` as **Layer Shifting** without changing the model class mapping.

The checkpoint must stay at `model/selected_highest_mask_mAP50.pt`. Its SHA256 is `33e2edae67c37d99c0138383d2a05db338c3f2857fd91a1c75f16f9a9b6f7ede`.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit. Upload a JPG or PNG image, then select **Analyze Image**. The app shows the original image, segmentation overlay, detection count, and each instance's class and confidence. Uploads are limited to 20 MB and 25 megapixels. The segmented PNG can be downloaded.

## Inference and FastAPI reference

Streamlit caches the model resource across interactions. Inference uses `imgsz=960`, `conf=0.25`, `iou=0.7`, and `verbose=False`. Masks, boxes, labels, and confidence are rendered with `result.plot()`.

## Streamlit Community Cloud

Commit and push `streamlit_app.py`, `static/streamlit.css`, `.streamlit/config.toml`, `requirements.txt`, and the model and static assets before creating the app.

Create a new app with these settings:

- **Repository:** `aamuros/AI-2_Deployment`
- **Branch:** `main`
- **Main file path (entrypoint):** `streamlit_app.py`

Community Cloud installs `requirements.txt` automatically. The checkpoint is about 92 MB, so the first model load may take time and the deployment needs enough memory for PyTorch and inference.

The original FastAPI implementation remains in `app.py` with its HTML, CSS, JavaScript, font, and icons in `static/`. Its dependencies remain in `requirements.txt` so it can still run as a reference with `python3 -m uvicorn app:app --reload`. The Streamlit styling is in `static/streamlit.css`; its page shape, footer reveal, and full-window drag prompt mirror the original design.

The drag-and-drop overlay uses the Picture, Clip 1, and Share 3 icons from [Streamline Pixel](https://www.streamlinehq.com/icons/pixel), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Local copies are in `static/icons/`.
