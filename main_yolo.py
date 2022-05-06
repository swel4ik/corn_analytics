from typing import List
from argparse import ArgumentParser, Namespace
import base64
import json
from time import gmtime, strftime
from fastapi import FastAPI, File
import uvicorn
from yolact.yolact import Yolact
from PIL import Image
from io import BytesIO
import uuid
import os
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


class CornStatistic(BaseModel):
    user_count: int
    predicted: int
    images: List[str]
    title: str


CORN_THRESHOLD = 0.99
KERNEL_THRESHOLD = 0.3


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


kernel_detector = torch.hub.load('yolo', 'custom', path='yolo/weights/best.pt',
                                 source='local')
kernel_detector.to(DEVICE)
logging.info('Kernel detector loaded')


linear_model = pickle.load(open('./yolact/weights/linear_kernel_1_2.sav', 'rb'))
logging.info('Linear model loaded')

app = FastAPI()


@app.post("/api/corn", status_code=200)
async def corn_count(files: List[str] = File(...)):
    _files = [file.split(',')[-1] for file in files]
    base64_images = [base64.b64decode(file) for file in _files]
    images = [np.array(Image.open(BytesIO(base64_img)).convert("RGB")) for base64_img in base64_images]
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

    data = np.array(sum(_kernels))

    if sum(_kernels) == 0:
        return {"Summary": 0}

    prediction = linear_model.predict(data.reshape(1, -1))[0]

    return {
        "Summary": prediction
    }


@app.post("/api/kernel/statistics", status_code=200)
async def kernel_statistics(statistics: CornStatistic):
    _statistics = statistics.dict()
    _id = str(uuid.uuid4())
    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    os.mkdir(f'./statistics/{time}_{_id}')

    _files = [file.split(',')[-1] for file in _statistics['images']]
    base64_images = [base64.b64decode(file) for file in _files]
    images = [Image.open(BytesIO(base64_img)).convert("RGB") for base64_img in base64_images]

    for num, img in enumerate(images):
        img.save(f'./statistics/{time}_{_id}/img_{num}.jpg')

    data = {
        'Real number of kernels': _statistics['user_count'],
        'Predicted number of kernels': _statistics['predicted']
    }

    with open(f"./statistics/{time}_{_id}/data.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return {
        'Success'
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Corn analysis server')
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app_log.info(
        '\n\n'
        'SERVER START TIME: {}'
        '\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    )

    uvicorn.run(app, host=args.ip, port=args.port)
