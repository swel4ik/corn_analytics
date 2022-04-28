import torch
from fastapi import FastAPI, File
import uvicorn
from yolact.yolact import Yolact
from PIL import Image
from io import BytesIO
import requests
import torch
from yolact_utils import detect_corn
import cv2
import numpy as np



use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
print(f'Device: {DEVICE}')


corn_detector = Yolact()
corn_detector.load_state_dict(torch.load('./yolact/weights/corn_detector_1351_50000.pth', map_location=DEVICE))
corn_detector.eval()
print('Corn detector loaded')


kernel_detector = torch.hub.load('yolov5', 'custom', path='yolov5/weights/yolov5m.pt',
                           source='local')
kernel_detector.to(DEVICE)
# kernel_detector.eval()
print('Kernel detector loaded')

app = FastAPI()


@app.get("/api/corn", status_code=200)
async def corn_count():
    response = requests.get('https://i.pinimg.com/originals/bc/d4/fe/bcd4fec1da42391c62d7449f6df9ced3.jpg')
    img = np.array(Image.open(BytesIO(response.content)).convert("RGB")) # Image.open(BytesIO(file)).convert("RGB")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corns = detect_corn(img, corn_detector)
    corn_kernels = {}
    for corn_num, corn in enumerate(corns):
        current_corn = cv2.cvtColor(corn, cv2.COLOR_BGR2RGB)

        kernel_detector_result = kernel_detector(current_corn)

        kernel_count = len(kernel_detector_result.pandas().xyxy[0])

        corn_kernels[f'Corn {corn_num}'] = kernel_count

    return {
        "Corns": len(corns),
        "Kernels": corn_kernels
            }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
