import numpy as np
import torch

def calculate_bounding_boxes(centroid_xy, landmark_wh):
    """
    Calculate the top-left and bottom-right coordinates of bounding boxes from centroids and dimensions.

    Args:
        centroid_xy (np.ndarray): Centroid coordinates of the landmarks as [x, y].
        landmark_wh (np.ndarray): Dimensions of the landmarks as [width, height].

    Returns:
        np.ndarray: An array of shape (n, 4), with each row containing the top-left and bottom-right coordinates of the bounding boxes.
    """
    # Calculate the top-left and bottom-right coordinates of the bounding boxes
    top_left = centroid_xy - landmark_wh / 2
    bottom_right = centroid_xy + landmark_wh / 2

    # Combine top-left and bottom-right into a single numpy array of shape (n, 4)
    bounding_boxes = np.hstack((top_left, bottom_right))


"""def postprocess(self, preds, img, orig_imgs):
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
        classes=self.args.classes,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    return results"""


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

