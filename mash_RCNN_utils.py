# -*- coding:utf-8 -*-
import numpy as np
import math
IMAGE_SIZE = 1024

def generate_anchors(scales, ratios, img_shape, feature_stride, anchor_stride):
    '''
    >>> RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    >>> RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    >>> import numpy as np
    >>> s, r = np.meshgrid(np.array(RPN_ANCHOR_SCALES), np.array(RPN_ANCHOR_RATIOS))
    >>> s
    array([[ 32,  64, 128, 256, 512],
           [ 32,  64, 128, 256, 512],
           [ 32,  64, 128, 256, 512]])
    >>> r
    array([[0.5, 0.5, 0.5, 0.5, 0.5],
           [1. , 1. , 1. , 1. , 1. ],
           [2. , 2. , 2. , 2. , 2. ]])
    >>> s = s.flatten()
    >>> s
    array([ 32,  64, 128, 256, 512,  32,  64, 128, 256, 512,  32,  64, 128,
           256, 512])
    >>> r.flatten()
    array([0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1. , 2. , 2. , 2. ,
           2. , 2. ])
    '''

    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    
    shifts_y = np.arange(0, img_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, img_shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    y_1 = boxes[..., 0]
    y_1 = np.where(y_1 < 0., 0., y_1)
    y_1 = np.expand_dims(y_1, 1)

    x_1 = boxes[..., 1]
    x_1 = np.where(x_1 < 0., 0., x_1)
    x_1 = np.expand_dims(x_1, 1)
    
    y_2 = boxes[..., 2]
    y_2_condition = y_2
    y_2_condition = np.where(y_2_condition > IMAGE_SIZE, 0., y_2_condition)
    y_2 = np.where(y_2 < 0. , 0., y_2_condition)
    y_2 = np.expand_dims(y_2, 1)

    x_2 = boxes[..., 3]
    x_2_condition = x_2
    x_2_condition = np.where(x_2_condition > IMAGE_SIZE, 0., x_2_condition)
    x_2 = np.where(x_2 < 0. , 0., x_2_condition)
    x_2 = np.expand_dims(x_2, 1)
    
    boxes = np.concatenate([y_1, x_1, y_2, x_2], axis=1)

    return boxes

def generate_pyramid_anchors(scales, ratios, 
                             feature_shapes, 
                             feature_strides,
                             anchor_stride):
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride,))

    return np.concatenate(anchors, axis=0)

def compute_backbone_shapes(image_shape, BACKBONE, BACKBONE_STRIDES):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    COMPUTE_BACKBONE_SHAPE = None
    if callable(BACKBONE):
        return COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in BACKBONE_STRIDES])

def normalize_box(box, img_shape):
    h, w = img_shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((box - shift), scale).astype(np.float32)


def get_anchor(image_shape, BACKBONE, BACKBONE_STRIDES,
               RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS, RPN_ANCHOR_STRIDE):

    backbone_shapes = compute_backbone_shapes(image_shape, BACKBONE, BACKBONE_STRIDES)
    anchor_cach = {}
    # Generate anchor
    a = generate_pyramid_anchors(
        RPN_ANCHOR_SCALES,
        RPN_ANCHOR_RATIOS,
        backbone_shapes,
        BACKBONE_STRIDES,
        RPN_ANCHOR_STRIDE)
    anchor_cach[tuple(image_shape)] = normalize_box(a, image_shape[:2])
    # ?? ymin, xmin?? ?????? ???????;;
    return anchor_cach[tuple(image_shape)]


# 내가 너무 어렵게 생각한것같다.,..