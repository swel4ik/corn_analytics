from fastapi import FastAPI, File
import uvicorn
from time import gmtime, strftime
from typing import List
import logging
from yolact.yolact import Yolact
from yolact.yolact_kernel import Yolact_kernel
from PIL import Image
from io import BytesIO
import pickle
import torch
from yolact_utils import detect_corn, cut_slices, predict, box_slice2global
import cv2
import numpy as np
from pydantic import BaseModel



class Corn(BaseModel):
    amounts: List[int]


app_log = logging.getLogger('ServerGlobalLog')
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
print(f'Device: {DEVICE}')


corn_detector = Yolact()
corn_detector.load_state_dict(torch.load('./yolact/weights/corn_detector_1351_50000.pth', map_location=DEVICE))
corn_detector.eval()
print('Corn detector loaded')

kernel_detector = Yolact_kernel()
kernel_detector.load_state_dict(torch.load('./yolact/weights/yolact_kernel_327_3600.pth', map_location=DEVICE))
kernel_detector.eval()
print('Kernel detector loaded')

linear_model = pickle.load(open('./yolact/weights/linear_kernel_1.sav', 'rb'))
print('Linear model loaded')

app = FastAPI()


@app.post("/api/corn", status_code=200)
async def corn_count(files: List[bytes] = File(...)):
    # response = requests.get('https://i.pinimg.com/originals/bc/d4/fe/bcd4fec1da42391c62d7449f6df9ced3.jpg')
    images = [np.array(Image.open(BytesIO(file)).convert("RGB")) for file in files]
    print(len(images))
    # img = np.array(Image.open(BytesIO(file)).convert("RGB"))  # Image.open(BytesIO(file)).convert("RGB")
    img = images[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img.save('out.png')
    corns = detect_corn(img, corn_detector)

    corn_kernels = {}
    for corn_num, corn in enumerate(corns):
        cv2.imwrite(f'{corn_num}_.jpg', corn)
        if corn.shape[0] > corn.shape[1]:
            corn = cv2.rotate(corn, cv2.cv2.ROTATE_90_CLOCKWISE)

        corn_slices, delta = cut_slices(corn, slices_number=3)

        kernel_detection_result = list(map(lambda slice_: predict(slice_, kernel_detector), corn_slices))

        boxes_in_slices = [detection[2] for detection in kernel_detection_result]
        sum = 0
        for bbox in boxes_in_slices:
            sum += len(bbox)

        kernels = len(box_slice2global(boxes=boxes_in_slices,
                                            slices=corn_slices,
                                            image=corn,
                                            delta=delta))

        corn_kernels[f'Corn {corn_num}'] = kernels

    return {
        "Corns": len(corns),
        "Kernels": corn_kernels
            }


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
    logging.basicConfig(filename='logs.txt', level=logging.INFO)
    app_log.info(
        '\n\n'
        'SERVER START TIME: {}'
        '\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)