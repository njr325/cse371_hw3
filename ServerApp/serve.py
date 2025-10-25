from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import torch
import cv2
import os
import numpy as np
from models.common import DetectPTBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

# ========== Flask App ==========
app = Flask(__name__)

# ========== YOLOv5 Model Config ==========
IMG_SIZE = (640, 640)
STRIDE = 32
MODEL_PATH = "./weights/yolov5m.pt"


def detect(image_bytes, weights=MODEL_PATH):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading YOLOv5 model...")
    model = DetectPTBackend(weights, device=device)
    print("Model loaded successfully!")

    # Load image from bytes
    im0 = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    im = letterbox(im0, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float().unsqueeze(0) / 255.0

    # Inference
    pred = model(im)
    det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    boxes = []
    if det is not None and len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            boxes.append({
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "score": float(conf),
                "cls": model.names[int(cls)]
            })
    return boxes


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get("data")
    if not data:
        return jsonify({"status": 400, "message": "No data received"}), 400

    image_bytes = base64.b64decode(data)
    boxes = detect(image_bytes)
    return jsonify({"status": 200, "detections": boxes})
    

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8008))
    app.run(debug=False, host='0.0.0.0', port=port)
