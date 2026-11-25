from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from src.config import (
    ALLOWED_EXTENSIONS,
    DATASETS,
    DEFAULT_HF_MODEL_REPO,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS,
    MAX_UPLOAD_BYTES,
    SCENARIOS,
)

# Guard against decompression bombs before decoding untrusted uploads.
Image.MAX_IMAGE_PIXELS = 25_000_000
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class InferenceError(ValueError):
    """Raised when uploaded media or model execution cannot proceed safely."""


@dataclass
class PredictionResult:
    predicted_code: str
    probabilities: dict[str, float]
    latency_ms: float
    original_rgb: Image.Image
    preprocessed_rgb: Image.Image
    original_size: tuple[int, int]
    cropped_size: tuple[int, int]
    model_path: Path


def crop_fundus_margins(
    rgb: Image.Image,
    threshold: int = 7,
    margin_ratio: float = 0.02,
    min_keep_ratio: float = 0.35,
) -> Image.Image:
    # Mirrors the exact V4 training pipeline threshold and safety margins.
    arr = np.asarray(rgb, dtype=np.uint8)
    height, width = arr.shape[:2]
    mask = arr.max(axis=2) > threshold
    row_active = mask.mean(axis=1) > 0.01
    col_active = mask.mean(axis=0) > 0.01
    if not row_active.any() or not col_active.any():
        return rgb

    y0 = int(np.argmax(row_active))
    y1 = int(height - np.argmax(row_active[::-1]))
    x0 = int(np.argmax(col_active))
    x1 = int(width - np.argmax(col_active[::-1]))

    pad_y = int(round((y1 - y0) * margin_ratio))
    pad_x = int(round((x1 - x0) * margin_ratio))
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)

    if (y1 - y0) < height * min_keep_ratio or (x1 - x0) < width * min_keep_ratio:
        return rgb
    return rgb.crop((x0, y0, x1, y1))


def preprocess_uploaded_image(
    raw_bytes: bytes,
    filename: str,
) -> tuple[Image.Image, Image.Image, tuple[int, int], np.ndarray]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise InferenceError(f"Format berkas tidak didukung. Gunakan berkas gambar ({allowed_str}).")

    if len(raw_bytes) == 0:
        raise InferenceError("Berkas yang diunggah kosong. Pilih citra fundus yang valid.")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise InferenceError("Ukuran berkas melebihi batas 15 MB. Perkecil resolusi gambar lalu coba kembali.")

    try:
        with Image.open(BytesIO(raw_bytes)) as src:
            original_rgb = ImageOps.exif_transpose(src).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InferenceError(
            "Berkas gagal dibaca sebagai citra yang valid atau resolusinya terlalu besar."
        ) from exc

    cropped = crop_fundus_margins(original_rgb)
    cropped_size = cropped.size
    letterboxed = ImageOps.pad(
        cropped,
        (IMAGE_SIZE, IMAGE_SIZE),
        color=(0, 0, 0),
        method=Image.Resampling.BILINEAR,
    )

    arr = np.asarray(letterboxed, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (arr - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0).astype(np.float32, copy=False)

    return original_rgb, letterboxed, cropped_size, batch


def _resolve_hf_repo_id() -> str:
    env_val = os.environ.get("HF_MODEL_REPO", "").strip()
    if env_val:
        return env_val
    try:
        secret_val = str(st.secrets.get("HF_MODEL_REPO", "")).strip()
        if secret_val:
            return secret_val
    except Exception:
        pass
    return DEFAULT_HF_MODEL_REPO.strip()


def get_model_path(dataset_key: str, scenario_key: str) -> Path:
    return DATASETS[dataset_key].onnx_dir / SCENARIOS[scenario_key].onnx_filename


def resolve_model_path(dataset_key: str, scenario_key: str) -> Path:
    local_path = get_model_path(dataset_key, scenario_key)
    if local_path.is_file():
        return local_path

    repo_id = _resolve_hf_repo_id()
    if not repo_id:
        raise InferenceError(
            f"Berkas model {local_path.name} tidak ditemukan di lokal dan HF_MODEL_REPO belum dikonfigurasi."
        )
    if not _HF_REPO_RE.match(repo_id):
        raise InferenceError("Format konfigurasi HF_MODEL_REPO tidak valid (gunakan format username/nama-repo).")

    remote_rel = f"{DATASETS[dataset_key].onnx_dir.name}/{SCENARIOS[scenario_key].onnx_filename}"
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_rel,
            repo_type="model",
        )
    except Exception as exc:
        raise InferenceError(
            f"Gagal mengunduh model {SCENARIOS[scenario_key].onnx_filename} dari Hugging Face ({repo_id})."
        ) from exc
    return Path(downloaded)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_onnx_session(model_path_str: str) -> ort.InferenceSession:
    # Keep a single ONNX session active in RAM to stay well under Streamlit Cloud free memory limits.
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path_str,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def run_prediction(
    raw_bytes: bytes,
    filename: str,
    dataset_key: str,
    scenario_key: str,
) -> PredictionResult:
    model_path = resolve_model_path(dataset_key, scenario_key)
    original_rgb, preprocessed_rgb, cropped_size, tensor = preprocess_uploaded_image(raw_bytes, filename)
    session = load_onnx_session(str(model_path))

    t0 = time.perf_counter()
    outputs = session.run(["logits"], {"input": tensor})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logits = np.asarray(outputs[0][0], dtype=np.float64)
    if not np.isfinite(logits).all():
        raise InferenceError("Model menghasilkan nilai numerik tidak valid untuk citra ini.")

    probs = _stable_softmax(logits)
    prob_map = {code: float(probs[idx]) for idx, code in enumerate(LABELS)}
    predicted_code = LABELS[int(np.argmax(probs))]

    return PredictionResult(
        predicted_code=predicted_code,
        probabilities=prob_map,
        latency_ms=latency_ms,
        original_rgb=original_rgb,
        preprocessed_rgb=preprocessed_rgb,
        original_size=original_rgb.size,
        cropped_size=cropped_size,
        model_path=model_path,
    )
