"""Streamlit deployment for the frozen FDM defect segmentation model."""

import base64
import logging
from copy import copy
from html import escape
from io import BytesIO
from math import floor
from pathlib import Path
from threading import Lock

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "selected_highest_mask_mAP50.pt"
FONT_PATH = BASE_DIR / "static" / "fonts" / "ppmondwest.woff2"
UPLOAD_ICON_PATH = BASE_DIR / "static" / "icons" / "upload.png"
PICTURE_ICON_PATH = BASE_DIR / "static" / "icons" / "picture.png"
ATTACHMENT_ICON_PATH = BASE_DIR / "static" / "icons" / "attachment.png"
SUPPORTED_CLASSES = {"Cracking", "Layer_Shifting", "Stringing", "Warping"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MODEL_LOCK = Lock()
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def load_model() -> YOLO:
    """Load and share one validated model across Streamlit reruns."""
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


def read_image(uploaded_file) -> Image.Image:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 20 MB limit.")
    try:
        source = open_image(BytesIO(raw))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("This file could not be opened as an image.") from None
    with source:
        if source.format not in {"JPEG", "PNG"}:
            raise ValueError("Upload a JPG or PNG image.")
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image resolution is too large (25 megapixels maximum).")
        try:
            return ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError):
            raise ValueError("This file could not be opened as an image.") from None


def analyze_image(model: YOLO, image: Image.Image) -> dict:
    with MODEL_LOCK:
        result = model.predict(
            source=image, imgsz=960, conf=0.25, iou=0.7, verbose=False,
        )[0]
        display_result = copy(result)
        names = result.names.items() if isinstance(result.names, dict) else enumerate(result.names)
        display_result.names = {
            class_id: name.replace("Layer_Shifting", "Layer Shifting")
            for class_id, name in names
        }
        # Ultralytics plots BGR arrays; Pillow and Streamlit expect RGB.
        plotted = Image.fromarray(display_result.plot()[..., ::-1])
        boxes = result.boxes
        detections = (
            [
                {
                    "class_name": result.names[int(class_id)].replace("Layer_Shifting", "Layer Shifting"),
                    "confidence": round(float(confidence), 4),
                }
                for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist())
            ]
            if boxes is not None else []
        )
    output = BytesIO()
    plotted.save(output, format="PNG")
    return {"segmented_image": output.getvalue(), "detections": detections}


def clear_results() -> None:
    st.session_state.pop("analysis", None)


def clear_upload() -> None:
    st.session_state.upload_generation += 1
    clear_results()


def add_styles() -> None:
    font = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
    upload_icon = base64.b64encode(UPLOAD_ICON_PATH.read_bytes()).decode("ascii")
    picture_icon = base64.b64encode(PICTURE_ICON_PATH.read_bytes()).decode("ascii")
    attachment_icon = base64.b64encode(ATTACHMENT_ICON_PATH.read_bytes()).decode("ascii")
    css = (BASE_DIR / "static" / "streamlit.css").read_text()
    css = css.replace("FONT_URL", f"data:font/woff2;base64,{font}")
    css = css.replace("UPLOAD_ICON_URL", f"data:image/png;base64,{upload_icon}")
    css = css.replace("PICTURE_ICON_URL", f"data:image/png;base64,{picture_icon}")
    css = css.replace("ATTACHMENT_ICON_URL", f"data:image/png;base64,{attachment_icon}")
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


def show_method() -> None:
    st.markdown(
        """
        <section class="method">
          <div><span class="section-index">THE METHOD</span><h2>From photo to<br>print insight.</h2>
          <p>A single image goes through the segmentation model. Each detected defect is marked on the output and listed with its confidence.</p></div>
          <div><span class="detail-label">SUPPORTED DEFECTS</span><ol class="defect-types">
            <li><span>01</span><strong>Cracking</strong><small>Gaps or splits in the printed surface</small></li>
            <li><span>02</span><strong>Layer shifting</strong><small>Layers displaced from their intended position</small></li>
            <li><span>03</span><strong>Stringing</strong><small>Thin strands of unwanted filament</small></li>
            <li><span>04</span><strong>Warping</strong><small>Lifted or curled edges</small></li>
          </ol></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_footer() -> None:
    st.markdown(
        """
        <footer class="site-footer"><div class="footer-top">
          <div><span class="footer-kicker">AI2 FINAL PROJECT</span><h2>People behind<br>the layers.</h2><p class="footer-submitted">Submitted to: Dr. Lysa Comia</p></div>
          <div class="footer-team"><span class="footer-kicker">FILAMEANT TEAM</span><ol class="footer-members">
            <li>Muros, Adrian</li><li>Sanchez, Jamin Ariane</li><li>Santos, Trent Joaqin</li><li>Taroma, Hideki Reiven</li>
          </ol></div></div><div class="footer-wordmark">Filameant</div></footer>
        """,
        unsafe_allow_html=True,
    )


def show_global_drop_overlay() -> None:
    st.html(
        """
        <div class="global-drop-overlay" id="globalDropOverlay" aria-hidden="true">
          <div class="global-drop-frame"><div class="global-drop-prompt">
            <div class="drop-file-icons" aria-hidden="true">
              <span class="picture-icon"></span><span class="attachment-icon"></span><span class="upload-icon"></span>
            </div>
            <p class="global-drop-title">Drop your print image.</p>
            <p class="global-drop-hint">JPG or PNG <span>·</span> One image <span>·</span> 20 MB max</p>
            <p class="global-drop-bottom">RELEASE ANYWHERE TO SELECT</p>
          </div></div>
        </div>
        <script>
        if (!window.fdmGlobalDropReady) {
          window.fdmGlobalDropReady = true;
          let dragDepth = 0;
          const overlay = () => document.getElementById('globalDropOverlay');
          const hasFiles = event => Array.from(event.dataTransfer?.types || []).includes('Files');
          const hide = () => {
            dragDepth = 0;
            overlay()?.classList.remove('is-visible');
            overlay()?.setAttribute('aria-hidden', 'true');
          };
          window.addEventListener('dragenter', event => {
            if (!hasFiles(event)) return;
            event.preventDefault();
            dragDepth += 1;
            overlay()?.classList.add('is-visible');
            overlay()?.setAttribute('aria-hidden', 'false');
          });
          window.addEventListener('dragover', event => {
            if (!hasFiles(event)) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
          });
          window.addEventListener('dragleave', event => {
            if (--dragDepth <= 0) hide();
          });
          window.addEventListener('drop', event => {
            if (!hasFiles(event)) return;
            hide();
            if (event.target instanceof Element
                && event.target.closest('[data-testid="stFileUploaderDropzone"]')) return;
            event.preventDefault();
            if (event.dataTransfer.files.length !== 1) return;
            const zone = document.querySelector('[data-testid="stFileUploaderDropzone"]');
            zone?.dispatchEvent(new DragEvent('drop', {
              bubbles: true, cancelable: true, dataTransfer: event.dataTransfer,
            }));
          });
          window.addEventListener('dragend', hide);
          window.addEventListener('blur', hide);
        }
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def main() -> None:
    st.set_page_config(page_title="Filameant", layout="wide", initial_sidebar_state="collapsed")
    st.session_state.setdefault("upload_generation", 0)
    add_styles()
    with st.container(key="page_surface"):
        with st.container(key="hero"):
            st.markdown(
                """
                <div class="hero-heading"><span class="eyebrow">FDM 3D PRINT INSPECTION</span>
                <h1>Inspect every layer.</h1>
                <p class="hero-subtitle">Upload a print photo to locate cracking, layer shifting, stringing, and warping.</p></div>
                <div class="upload-heading"><span class="section-index">IMAGE INPUT</span><h2>Start with a print photo.</h2></div>
                """, unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Drop your image here · JPG or PNG, up to 20 MB",
                type=["jpg", "jpeg", "png"], accept_multiple_files=False,
                on_change=clear_results, key=f"uploaded_image_{st.session_state.upload_generation}",
            )

            image = None
            clicked = False
            if uploaded is not None:
                try:
                    image = read_image(uploaded)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    with st.container(key="selected_upload"):
                        st.image(image)
                        st.button("×", key="remove_upload", help="Remove uploaded image", on_click=clear_upload)
                        filename = escape(uploaded.name)
                        size_mb = len(uploaded.getvalue()) / 1024 / 1024
                        st.markdown(
                            f'<div class="selected-file"><strong>{filename}</strong>'
                            f'<span>{size_mb:.1f} MB · Ready to analyze</span></div>',
                            unsafe_allow_html=True,
                        )
                        clicked = st.button("Analyze Image", key="analyze_image", width="stretch")

        if clicked and image is not None:
            clear_results()
            try:
                with st.spinner("Loading segmentation model…"):
                    model = load_model()
            except Exception:
                LOGGER.exception("Could not load the segmentation checkpoint")
                st.error("The segmentation model is unavailable. Check the checkpoint and deployment logs.")
            else:
                try:
                    with st.spinner("Analyzing image…"):
                        st.session_state.analysis = analyze_image(model, image)
                except Exception:
                    LOGGER.exception("Image inference failed")
                    st.session_state.pop("analysis", None)
                    st.error("Image analysis failed. Please try another image.")

        analysis = st.session_state.get("analysis")
        if analysis is not None and image is not None:
            with st.container(key="results"):
                count = len(analysis["detections"])
                st.markdown(
                    '<div class="section-heading"><div><span class="section-index">ANALYSIS</span>'
                    '<h2>Inspection result</h2></div>'
                    f'<div class="result-summary"><strong>{count}</strong><span>defects<br>found</span></div></div>',
                    unsafe_allow_html=True,
                )
                detail, _ = st.columns([4, 1])
                with detail:
                    st.markdown('<h3 class="detection-heading">Detected defects</h3>', unsafe_allow_html=True)
                    if not analysis["detections"]:
                        st.write("No supported FDM defect was detected in this image.")
                    for index, detection in enumerate(analysis["detections"], start=1):
                        name = escape(detection["class_name"])
                        confidence = floor(detection["confidence"] * 100 + 0.5)
                        st.markdown(
                            f'<div class="detection-row"><span class="number">{index:02d}</span>'
                            f'<strong class="name">{name}</strong><span class="confidence">{confidence}% '
                            '<small>confidence</small></span></div>', unsafe_allow_html=True,
                        )
                original_col, result_col = st.columns(2, gap="medium")
                with original_col:
                    st.image(image, width="stretch")
                    st.markdown('<div class="image-label">Input image <span>01 / ORIGINAL</span></div>', unsafe_allow_html=True)
                with result_col:
                    st.image(analysis["segmented_image"], width="stretch")
                    st.markdown('<div class="image-label">Segmented image <span>02 / MODEL OUTPUT</span></div>', unsafe_allow_html=True)
                    st.download_button("Download", analysis["segmented_image"], "fdm-segmentation.png", "image/png", width="stretch")

                if clicked:
                    # Match the original page's scroll to a newly completed analysis.
                    st.html(
                        """
                        <script>
                        setTimeout(() => {
                          const results = document.querySelector('.st-key-results');
                          const scroller = document.querySelector('[data-testid="stMain"]');
                          if (results && scroller) {
                            const top = results.getBoundingClientRect().top
                              - scroller.getBoundingClientRect().top + scroller.scrollTop;
                            const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
                            scroller.scrollTo({ top, behavior: reducedMotion ? 'instant' : 'smooth' });
                          }
                        }, 150);
                        </script>
                        """,
                        unsafe_allow_javascript=True,
                    )

        with st.container(key="method"):
            show_method()
    show_footer()
    show_global_drop_overlay()


if __name__ == "__main__":
    main()
