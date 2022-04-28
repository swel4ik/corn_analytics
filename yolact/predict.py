from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from tqdm import tqdm
from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cpu()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=0.99)
        cfg.rescore_bbox = save
    return t


def evalimage(net: Yolact, path: str, save_path: str = None):
    frame = torch.from_numpy(cv2.imread(path)).cpu().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)

# image_path = 'D:/corn_yolo/train/DSC03570.jpg'

net = Yolact()
net.load_weights('D:/yolact_base_2222_60000.pth')

net.cpu()
net.eval()

import glob
# images = glob.glob('D:/corn_yolo/train_coco/JPEGImages/*')
images = glob.glob('C:/Users/zador/Desktop/lae/*')
dst_dir = 'D:/kernel_database/train'

for img_path in tqdm(images):
    img = cv2.imread(img_path)
    image_name = img_path.split('/')[-1].split('\\')[-1]
    try:
        frame = torch.from_numpy(img).cpu().float()
    except TypeError:
        print('a')
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    _, _, box, mask = prep_display(preds, frame, None, None, undo_transform=False)
    for i, bbox in enumerate(box):
        try:
            mask_to_save = mask.permute(1, 2, 0).cpu().detach().numpy()
            x1, y1, x2, y2 = box[i]
            # img_to_save = img * np.expand_dims(mask_to_save[:, :, i], axis=2)
        except TypeError:
            print('a')

        d_x_l = int(0.008 * img.shape[1])
        d_x_r = int(0.008 * img.shape[1])
        d_y_l = int(0.006 * img.shape[0])
        d_y_r = int(0.006 * img.shape[0])

        if x1 - d_x_l < 0:
            d_x_l = 0
        if x2 + d_x_r > img.shape[1]:
            d_x_r = 0

        if y1-d_y_l < 0:
            d_y_l = 0
        if y2 + d_y_r > img.shape[0]:
            d_y_r = 0

        img_to_save = img[y1-d_y_l:y2+d_y_r, x1-d_x_l:x2+d_x_r]
        cv2.imwrite(f'{dst_dir}/{i}_{image_name}', img_to_save)