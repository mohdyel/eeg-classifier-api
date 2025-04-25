"""
EEG spectrogram classifier + REST bridge.
Start with:      python eeg_flet_server.py
Endpoints:
  POST /predict         { "parquet_path": "/abs/or/relative/path.parquet" }
  POST /predict_bytes   { "file_name": "...", "file_b64": "iVBORw0…" }

Both return:
  { "class_": "LRDA", "image_b64": "<PNG‑base64>", "file": "<original name>" }
"""

import base64, io, os, threading

import cv2
import flet as ft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
import keras, keras_cv, tensorflow as tf
import os


# ------------------------------------------------------------------ config
class CFG:
    image_size = [400, 300]                # (height, width)
    num_classes = 6
    preset = "efficientnetv2_b2_imagenet"  # KerasCV preset
    weights_path = "lib/99.keras"          # <- adjust if needed
    label2name = {0: "Seizure", 1: "LPD", 2: "GPD",
                  3: "LRDA", 4: "GRDA", 5: "Other"}


# --------------------------------------------------------------- utilities
def _read_parquet(maybe_path_or_buf):
    """Accepts a filesystem path *or* a BytesIO and returns a pandas DF."""
    return pd.read_parquet(maybe_path_or_buf)


def convert_parquet_to_spectrogram(source, target_shape=(400, 300)):
    """
    source: path (str / os.PathLike) **or** BytesIO containing parquet bytes
    returns: (H,W) float32 array, normalized log‑spectrogram
    """
    df = _read_parquet(source)

    # drop first column (time), transpose → (freq, time)
    spec = df.fillna(0).values[:, 1:].T.astype(np.float32)

    # resize freq axis if needed
    cur_h, cur_w = spec.shape
    if cur_h != target_shape[0]:
        spec = cv2.resize(spec, (cur_w, target_shape[0]))

    # fix / crop / pad time axis
    if cur_w < target_shape[1]:
        spec = np.pad(spec, ((0, 0), (0, target_shape[1] - cur_w)))
    else:
        spec = spec[:, :target_shape[1]]

    # log‐compression & normalization
    spec = np.clip(spec, np.exp(-4.0), np.exp(8.0))
    spec = np.log(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec


def spectrogram_png_b64(spec2d, pred_class):
    """Make a PNG plot in memory and return base‑64 text."""
    plt.figure(figsize=(7, 6))
    plt.imshow(spec2d, cmap="viridis", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title(f"Predicted: {pred_class}", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ------------------------------------------------------------- load model
model = keras_cv.models.ImageClassifier.from_preset(
    CFG.preset, num_classes=CFG.num_classes)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=keras.losses.KLDivergence())
model.load_weights(CFG.weights_path)
print("✔ model loaded")


def run_inference(spec2d):
    img = np.stack([spec2d, spec2d, spec2d], -1)[None]  # (1,H,W,3)
    probs = model.predict(img, verbose=0)
    cls_idx = int(np.argmax(probs, -1)[0])
    return CFG.label2name[cls_idx]


# ------------------------------------------------------------------- REST
api = Flask(__name__)
CORS(api, resources={r"/predict*": {"origins": "*"}}, allow_headers="Content-Type")


@api.route("/predict", methods=["POST"])
def predict_path():
    parquet_path = request.get_json(force=True).get("parquet_path")
    if not parquet_path or not os.path.exists(parquet_path):
        return jsonify(error="parquet_path missing or does not exist"), 400
    spec = convert_parquet_to_spectrogram(parquet_path, tuple(CFG.image_size))
    cls = run_inference(spec)
    img_b64 = spectrogram_png_b64(spec, cls)
    return jsonify(class_=cls, image_b64=img_b64, file=os.path.basename(parquet_path))


@api.route("/predict_bytes", methods=["POST"])
def predict_bytes():
    data = request.get_json(force=True)
    b64 = data.get("file_b64")
    if not b64:
        return jsonify(error="file_b64 missing"), 400

    try:
        raw = base64.b64decode(b64)
        spec = convert_parquet_to_spectrogram(io.BytesIO(raw), tuple(CFG.image_size))
    except Exception as e:
        return jsonify(error=f"decode/read failed: {e}"), 400

    cls = run_inference(spec)
    img_b64 = spectrogram_png_b64(spec, cls)
    return jsonify(class_=cls, image_b64=img_b64, file=data.get("file_name", ""))


def _start_api():
    port = int(os.environ.get("PORT", 8000))
    api.run(host="0.0.0.0", port=port, debug=False)


# -------------------------------------------------------------- Flet page
def main(page: ft.Page):
    page.title = "EEG model server"
    page.add(
        ft.Text("Ready.\n"
                "• /predict expects {parquet_path: ...}\n"
                "• /predict_bytes expects {file_name, file_b64}"))


# --------------------------------------------------------------- bootstrap
app = api
if __name__ == "__main__":
    threading.Thread(target=_start_api, daemon=True).start()
    ft.app(target=main, view=None, port=8550)
