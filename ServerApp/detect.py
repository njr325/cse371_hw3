from pathlib import Path
import os
import argparse
import cv2
import numpy as np

import torch
# import torch.backends.cudnn as cudnn
from models.common import DetectPTBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors, save_one_box

# ========= Configuration related to the Yolov5 ========≠
IMG_SIZE = (640, 640)
STRIDE = 32

def load_model(weights, device):
    model = torch.load(weights, map_location=device)
    return model['model'].float()

def detect(image_path,
           weights):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference on device: ", device)
    # =========== Load Model ===========
    print(f"Loading model from {weights}")
    model = DetectPTBackend(weights, device=device)
    print(f"Loading model completed.")
    
    # =========== Load Image ===========
    im0 = cv2.imread(image_path)
    assert im0 is not None, f'Image Not Found {image_path}'
    
    # =========== Preprocess Image ===========
    im = letterbox(im0, IMG_SIZE[0], stride=32, auto=True)[0] # Resize and padding
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im) # (3, 640, 640)
    im = torch.from_numpy(im).to(device).float().unsqueeze(0) # Convert image to tensor
    im /= 255  # 0 - 255 to 0.0 - 1.0
    
    ### =========== Inference ===========
    pred = model(im) # (1, 25200, 85) (center x, center y, width, height, conf, 80 class prob)
    # print(pred[0].numpy().flatten()[:10])
    
    ### =========== Post-processing ===========
    det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)[0]  # (N, 6)  (x1, y1, x2, y2, conf, cls)
    
    
    ### ================================================
    ### =========== (Optional) Visualization ===========
    names = model.names # get object category names
    annotator = Annotator(im0, line_width=2, example=str(names))
    s = '%gx%g ' % im.shape[2:]  # print string
    if len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() # Scale the bounding box to original image
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()

    cv2.imshow("detection", im0)
    cv2.waitKey()  # 1 millisecond


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='the image to detect')
    parser.add_argument('--weights', type=str, default="./models/yolov5s.pt", help='model path')
    opt = parser.parse_args()
    detect(**vars(opt))