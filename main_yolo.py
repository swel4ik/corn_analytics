import torch
from typing import List
from time import gmtime, strftime
from fastapi import FastAPI, File
import uvicorn
from yolact.yolact import Yolact
from PIL import Image
from io import BytesIO
import requests
import logging
import torch
from yolact_utils import detect_corn
import cv2
import numpy as np
from pydantic import BaseModel
import pickle
from metrics.yolo_metric_eval import get_kernels_count


class Corn(BaseModel):
    amounts: List[int]


CORN_THRESHOLD = 0.99
KERNEL_THRESHOLD = 0.4


app_log = logging.getLogger('ServerGlobalLog')
logging.basicConfig(filename='./logs/logs.txt', level=logging.INFO)

# Load models

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
logging.info(f'Device: {DEVICE}')


corn_detector = Yolact()
corn_detector.load_state_dict(torch.load('./yolact/weights/corn_detector_1351_50000.pth', map_location=DEVICE))
corn_detector.eval()
logging.info('Corn detector loaded')


kernel_detector = torch.hub.load('yolov5', 'custom', path='yolov5/weights/yolov5m.pt',
                           source='local')
kernel_detector.to(DEVICE)
logging.info('Kernel detector loaded')


linear_model = pickle.load(open('./yolact/weights/linear_kernel_1.sav', 'rb'))
logging.info('Linear model loaded')

app = FastAPI()


@app.post("/api/corn", status_code=200)
async def corn_count(files: List[bytes] = File(...)):
    images = [np.array(Image.open(BytesIO(file)).convert("RGB")) for file in files]
    result = {}
    for img_num, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        corns = detect_corn(img, corn_detector, th=CORN_THRESHOLD)
        corn_kernels = {}

        for corn_num, corn in enumerate(corns):
            current_corn = cv2.cvtColor(corn, cv2.COLOR_BGR2RGB)
            kernel_detector_result = kernel_detector(current_corn).pandas().xyxy[0]
            kernel_count = get_kernels_count(kernel_detector_result, th=KERNEL_THRESHOLD)
            corn_kernels[f'Corn {corn_num}'] = kernel_count

        result[f'Image {img_num}'] = corn_kernels

    return result


@app.post("/api/kernel/count", status_code=200)
async def kernel_count(corn: Corn):
    kernels = corn.dict()
    _kernels = kernels['amounts']
    _kernels = [int(num) for num in _kernels]

    if len(_kernels) == 1:
        _kernels.append(_kernels[0])
        _kernels.append(_kernels[0])

    elif len(_kernels) == 2:
        _kernels.append(np.mean(_kernels))

    prediction = linear_model.predict([_kernels])[0]

    return {
        "Summary": prediction
    }


if __name__ == "__main__":
    app_log.info(
        '\n\n'
        'SERVER START TIME: {}'
        '\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
