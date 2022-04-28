import torch
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm


def get_kernels_count(predict, th=0.):
    return len(predict[predict['confidence'] > th])


images_dir = 'D:/all_corn/kernel_database/valid'

kernel_detector = torch.hub.load('../yolov5', 'custom', path='yolov5/weights/yolov5m.pt',
                                 source='local')  # or yolov5m, yolov5l, yolov5x, custom

# Images
preds = []
gt = []
for file in tqdm(glob.glob('D:/all_corn/kernel_database/valid/*.json')):
    name = file.split('/')[-1].split("\\")[-1].split('.')[0]
    try:
        img = cv2.imread(f'D:/all_corn/kernel_database/valid/{name}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = cv2.imread(f'D:/all_corn/kernel_database/valid/{name}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(file, "r") as read_file:
        data = json.load(read_file)

    # Inference
    pred = kernel_detector(img).pandas().xyxy[0]
    th = 0.4
    kernels = get_kernels_count(pred, th)
    preds.append(kernels)
    gt.append(len(data['shapes']))


preds = np.array(preds)
gt = np.array(gt)
res = np.mean(np.abs(preds - gt))

mae = np.abs(preds - gt)
res_2 = np.round(np.mean(mae / gt) * 100., 3)
print('Threshold:', th)
print(res)
print(f'{res_2}%')

