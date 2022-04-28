import torch
import cv2
import glob
import json
import numpy as np
from yolact_utils import detect_corn, cut_slices, predict, box_slice2global
from yolact.yolact_kernel import Yolact_kernel
from tqdm import tqdm


DEVICE='cpu'
def get_kernels_count(predict, th=0.):
    return len(predict[predict['confidence'] > th])


images_dir = 'D:/all_corn/kernel_database/valid'

kernel_detector = Yolact_kernel()
kernel_detector.load_state_dict(torch.load('../yolact/weights/yolact_kernel_327_3600.pth', map_location=DEVICE))
kernel_detector.eval()
print('Kernel detector loaded')

# Images
th = 0.5
preds = []
gt = []
for file in tqdm(glob.glob('D:/all_corn/kernel_database/valid/*.json')):
    name = file.split('/')[-1].split("\\")[-1].split('.')[0]
    try:
        img = cv2.imread(f'D:/all_corn/kernel_database/valid/{name}.png')
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = cv2.imread(f'D:/all_corn/kernel_database/valid/{name}.jpg')
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(file, "r") as read_file:
        data = json.load(read_file)

    corn_slices, delta = cut_slices(img, slices_number=3)

    kernel_detection_result = list(map(lambda slice_: predict(slice_, kernel_detector, th=th), corn_slices))

    boxes_in_slices = [detection[2] for detection in kernel_detection_result]
    sum = 0
    for bbox in boxes_in_slices:
        sum += len(bbox)

    kernels = len(box_slice2global(boxes=boxes_in_slices,
                                   slices=corn_slices,
                                   image=img,
                                   delta=delta))
    gt.append(len(data['shapes']))

    preds.append(kernels)

preds = np.array(preds)
gt = np.array(gt)
res = np.mean(np.abs(preds - gt))

mae = np.abs(preds - gt)
res_2 = np.round(np.mean(mae / gt) * 100., 3)
print('Threshold:', th)
print(res)
print(f'{res_2}%')

