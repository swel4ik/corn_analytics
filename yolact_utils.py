from yolact.layers.output_utils import postprocess, undo_image_transformation
import torch
from yolact.data import cfg, set_cfg, set_dataset
from yolact.utils.augmentations import BaseTransform, FastBaseTransform, Resize
import numpy as np

NMS_THRESH = 0.7

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str='', th=0.5):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_cpu = torch.Tensor(img_numpy).cpu()
    else:
        img_cpu = img / 255.0
        h, w, _ = img.shape

    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=th)
    cfg.rescore_bbox = save
    return t


def detect_corn(inp, net, th=0.5):
    classes, scores, box, mask = predict(inp, net, th=th)
    corns = []
    for i, bbox in enumerate(box):

        x1, y1, x2, y2 = box[i]
        d_x_l = int(0.008 * inp.shape[1])
        d_x_r = int(0.008 * inp.shape[1])
        d_y_l = int(0.006 * inp.shape[0])
        d_y_r = int(0.006 * inp.shape[0])

        if x1 - d_x_l < 0:
            d_x_l = 0
        if x2 + d_x_r > inp.shape[1]:
            d_x_r = 0

        if y1-d_y_l < 0:
            d_y_l = 0
        if y2 + d_y_r > inp.shape[0]:
            d_y_r = 0

        corn = inp[y1 - d_y_l:y2 + d_y_r, x1 - d_x_l:x2 + d_x_r]
        corns.append(corn)

    return corns


def predict(image: np.array, net, th=0.5):
    frame = torch.from_numpy(image).cpu().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    classes, scores, box, mask = prep_display(preds, frame, None, None, undo_transform=False, th=th)
    return classes, scores, box, mask


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #     scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_classic(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #     scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def cut_slices(image: np.array, slices_number: int = 3):
    slices = []
    delta = 0

    for i in range(0, image.shape[1], image.shape[1] // slices_number):
        slices.append(image[:, i - delta:i + image.shape[1] // slices_number, :])
        delta = (image.shape[1] // 8) // slices_number

    return slices, delta


def box_slice2global(boxes: list, slices: list, image: np.array, delta):
    global_boxes = []

    for slice_num, slice_ in enumerate(slices[0:3]):
        for box_num, bbox in enumerate(boxes[slice_num]):
            x1, y1, x2, y2 = bbox
            if slice_num != 0:

                x1 += slice_num * image.shape[1] // 3 - delta
                x2 += slice_num * image.shape[1] // 3 - delta

            global_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

    return nms(np.asarray(global_boxes), thresh=NMS_THRESH)