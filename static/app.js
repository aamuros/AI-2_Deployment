const form = document.getElementById("analyzeForm");
const fileInput = document.getElementById("fileInput");
const chooseButton = document.getElementById("chooseButton");
const dropZone = document.getElementById("dropZone");
const globalDropOverlay = document.getElementById("globalDropOverlay");
const uploadPreviewWrap = document.getElementById("uploadPreviewWrap");
const uploadPreview = document.getElementById("uploadPreview");
const removeFileButton = document.getElementById("removeFileButton");
const fileName = document.getElementById("fileName");
const fileHint = document.getElementById("fileHint");
const formStatus = document.getElementById("formStatus");
const results = document.getElementById("results");
const originalImage = document.getElementById("originalImage");
const segmentedImage = document.getElementById("segmentedImage");
const resultCount = document.getElementById("resultCount");
const detectionList = document.getElementById("detectionList");
const downloadResult = document.getElementById("downloadResult");

const allowedTypes = new Set(["image/jpeg", "image/png"]);
const maxUploadBytes = 20 * 1024 * 1024;
let selectedFile = null;
let previewUrl = null;
let busy = false;
let dragDepth = 0;

function setStatus(message, isError = false) {
  formStatus.textContent = message;
  formStatus.classList.toggle("is-error", isError);
}

function setBusy(value) {
  busy = value;
  form.setAttribute("aria-busy", String(value));
  chooseButton.disabled = value;
  chooseButton.type = selectedFile ? "submit" : "button";
  chooseButton.textContent = value ? "Analyzing…" : selectedFile ? "Analyze print" : "Upload file";
  removeFileButton.disabled = value;
}

function selectFile(file) {
  if (busy || !file) return;
  if (!allowedTypes.has(file.type)) {
    clearSelection();
    setStatus("Upload a JPG or PNG image.", true);
    return;
  }
  if (file.size > maxUploadBytes) {
    clearSelection();
    setStatus("Image exceeds the 20 MB limit.", true);
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  selectedFile = file;
  uploadPreview.src = previewUrl;
  uploadPreviewWrap.hidden = false;
  dropZone.classList.add("has-file");
  fileName.textContent = file.name;
  fileHint.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to analyze`;
  results.hidden = true;
  setStatus("");
  setBusy(false);
}

function clearSelection() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  fileInput.value = "";
  uploadPreviewWrap.hidden = true;
  uploadPreview.removeAttribute("src");
  dropZone.classList.remove("has-file");
  fileName.textContent = "Drop your image here";
  fileHint.textContent = "JPG or PNG, up to 20 MB";
  results.hidden = true;
  setBusy(false);
}

chooseButton.addEventListener("click", () => {
  if (!selectedFile) fileInput.click();
});
removeFileButton.addEventListener("click", () => {
  clearSelection();
  setStatus("");
  chooseButton.focus();
});
fileInput.addEventListener("change", () => {
  selectFile(fileInput.files[0]);
  fileInput.value = "";
});

function hasDraggedFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

function hideDropOverlay() {
  dragDepth = 0;
  globalDropOverlay.classList.remove("is-visible");
  globalDropOverlay.setAttribute("aria-hidden", "true");
}

window.addEventListener("dragenter", (event) => {
  if (busy || !hasDraggedFiles(event)) return;
  event.preventDefault();
  dragDepth += 1;
  globalDropOverlay.classList.add("is-visible");
  globalDropOverlay.setAttribute("aria-hidden", "false");
});

window.addEventListener("dragover", (event) => {
  if (!hasDraggedFiles(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = busy ? "none" : "copy";
});

window.addEventListener("dragleave", (event) => {
  if (!dragDepth) return;
  dragDepth -= 1;
  if (dragDepth === 0 || event.clientX <= 0 || event.clientY <= 0 || event.clientX >= innerWidth || event.clientY >= innerHeight) {
    hideDropOverlay();
  }
});

window.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;
  if (!files?.length && !dragDepth) return;
  event.preventDefault();
  hideDropOverlay();
  if (busy) return;
  if (!files || files.length !== 1) {
    clearSelection();
    setStatus("Drop one JPG or PNG image at a time.", true);
    return;
  }
  selectFile(files[0]);
  form.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
});

window.addEventListener("dragend", hideDropOverlay);
window.addEventListener("blur", hideDropOverlay);

function renderDetections(detections) {
  detectionList.replaceChildren();
  if (!detections.length) {
    const empty = document.createElement("p");
    empty.className = "empty-result";
    empty.textContent = "No supported FDM defect was detected in this image.";
    detectionList.append(empty);
    return;
  }

  detections.forEach((detection, index) => {
    const row = document.createElement("div");
    row.className = "detection-row";
    const number = document.createElement("span");
    number.className = "row-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("strong");
    name.className = "defect-name";
    name.textContent = detection.class_name;
    const confidence = document.createElement("span");
    confidence.className = "confidence";
    const confidenceValue = document.createElement("span");
    confidenceValue.className = "confidence-value";
    confidenceValue.textContent = `${Math.round(detection.confidence * 100)}%`;
    const confidenceLabel = document.createElement("span");
    confidenceLabel.className = "confidence-label";
    confidenceLabel.textContent = "confidence";
    confidence.append(confidenceValue, confidenceLabel);
    row.append(number, name, confidence);
    detectionList.append(row);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile || busy) return;

  setBusy(true);
  setStatus("Running segmentation. This can take a moment.");
  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Image analysis failed. Please try again.");

    originalImage.src = previewUrl;
    segmentedImage.src = data.segmented_image;
    resultCount.textContent = data.total_detections;
    renderDetections(data.detections);
    downloadResult.href = data.segmented_image;
    results.hidden = false;
    setStatus("Analysis complete.");
    results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  } catch (error) {
    setStatus(error.message || "Image analysis failed. Please try again.", true);
  } finally {
    setBusy(false);
  }
});
