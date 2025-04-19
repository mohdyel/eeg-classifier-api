import base64, io, os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_cv
import keras
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
class CFG:
    image_size = [400, 300]
    num_classes = 6
    preset = "efficientnetv2_b2_imagenet"
    weights_path = "lib/99.keras"
    label2name = {0: "Seizure", 1: "LPD", 2: "GPD", 3: "LRDA", 4: "GRDA", 5: "Other"}

model = keras_cv.models.ImageClassifier.from_preset(
    CFG.preset, num_classes=CFG.num_classes)
model.load_weights(CFG.weights_path)

def convert_parquet_to_spectrogram(source, target_shape=(400, 300)):
    df = pd.read_parquet(source)
    spec = df.fillna(0).values[:, 1:].T.astype(np.float32)
    cur_h, cur_w = spec.shape
    if cur_h != target_shape[0]:
        spec = cv2.resize(spec, (cur_w, target_shape[0]))
    if cur_w < target_shape[1]:
        spec = np.pad(spec, ((0, 0), (0, target_shape[1] - cur_w)))
    else:
        spec = spec[:, :target_shape[1]]
    spec = np.clip(spec, np.exp(-4.0), np.exp(8.0))
    spec = np.log(spec)
    return (spec - spec.mean()) / (spec.std() + 1e-6)

@app.route("/predict_bytes", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        b64 = data.get("file_b64")
        raw = base64.b64decode(b64)
        spec = convert_parquet_to_spectrogram(io.BytesIO(raw))
        img = np.stack([spec, spec, spec], -1)[None]
        probs = model.predict(img, verbose=0)
        cls_idx = int(np.argmax(probs, -1)[0])
        return {"class_": CFG.label2name[cls_idx]}
    except Exception as e:
        return {"error": str(e)}, 500
