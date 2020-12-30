# -*- coding:utf-8 -*-
from absl import flags
from mash_RCNN_utils import *
from model import *

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_integer("img_size", 1024, "Image size (Original is 416, but to use pre-trained)")

flags.DEFINE_integer("batch_size", 8, "Batch size")

flags.DEFINE_multi_integer("RPN_ANCHOR_SCALES", (32, 64, 128, 256, 512), "")

flags.DEFINE_multi_float("RPN_ANCHOR_RATIOS", [0.5, 1, 2], "")

flags.DEFINE_multi_integer("BACKBONE_STRIDES", [4, 8, 16, 32, 64], "")

flags.DEFINE_integer("RPN_ANCHOR_STRIDE", 1, "")


FLAGS = flags.FLAGS
FLAGS(sys.argv)

def func_(img_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.

    return img

def read_label(file, batch_size):
    # https://github.com/jmpap/YOLOV2-Tensorflow-2.0/blob/master/Yolo_V2_tf_eager.ipynb

    anchor_count = 5
    responsibleGrid = np.zeros([FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, 5, 25])

    detector_mask = np.zeros((FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, anchor_count, 1))
    matching_true_boxes = np.zeros((FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, anchor_count, 5))

    full_target_grid = []
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        traget_grid = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]

            xmin = (float(line.split(',')[0]))
            xmax = (float(line.split(',')[2]))
            ymin = (float(line.split(',')[1]))
            ymax = (float(line.split(',')[3]))
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])
            classes = int(line.split(',')[6])

            xmin = xmin
            xmax = xmax

            ymin = ymin
            ymax = ymax

            if xmax > xmin and ymax > ymin:
                x = (xmin + xmax) * 0.5
                x = x / FLAGS.img_size / FLAGS.output_size
                y = (ymin + ymax) * 0.5
                y = y / FLAGS.img_size / FLAGS.output_size
                grid_x = int(np.floor(x))
                grid_y = int(np.floor(y))
                if grid_x < FLAGS.output_size and grid_y < FLAGS.output_size:
                    w = (xmax - xmin) / FLAGS.img_size / FLAGS.output_size
                    h = (ymax - ymin) / FLAGS.img_size / FLAGS.output_size

                    boxData = [x, y, w, h]
                    best_box = 0
                    best_anchor = 0
                    for i in range(anchor_count):
                        intersect = np.minimum(w, ANCHORS_box[i, 0]) * np.minimum(h, ANCHORS_box[i, 1])
                        union = ANCHORS_box[i, 0] * ANCHORS_box[i, 1] + (w * h) - intersect
                        iou = intersect / union
                        if iou > best_box:
                            best_box = iou
                            best_anchor = i

                    responsibleGrid[b][grid_x][grid_y][best_anchor][classes] = 1.    # class
                    responsibleGrid[b][grid_x][grid_y][best_anchor][21:25] = boxData    # box
                    responsibleGrid[b][grid_x][grid_y][best_anchor][FLAGS.num_classes] = 1. # confidence

                    traget_grid.append([classes, 1, x, y, w, h])

        full_target_grid.append(traget_grid)

    responsibleGrid = np.array(responsibleGrid, dtype=np.float32)
    traget_grid = np.array(traget_grid, dtype=np.float32)

    return responsibleGrid, full_target_grid

def main():

    model = ResNet_FPN(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                       architecture="resnet101")
    model.summary()

    image_list = os.listdir(FLAGS.tr_img_path)
    image_list = [FLAGS.tr_img_path + '/' + data for data in image_list]

    data = tf.data.Dataset.from_tensor_slices(image_list)
    data = data.shuffle(len(image_list))
    data = data.map(func_)
    data = data.batch(FLAGS.batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    anchors = get_anchor(np.array([1024, 1024, 3]),
                         "resnet101",
                         FLAGS.BACKBONE_STRIDES,
                         FLAGS.RPN_ANCHOR_SCALES,
                         FLAGS.RPN_ANCHOR_RATIOS,
                         FLAGS.RPN_ANCHOR_STRIDE)
    anchors = np.broadcast_to(anchors, (FLAGS.batch_size,) + anchors.shape)


    it = iter(data)
    for i in range(10):
        img = next(it)

        rpn_class_logits, rpn_class, rpn_bbox = model(img)

if __name__ == "__main__":
    main()

# image_shape, BACKBONE, BACKBONE_STRIDES,
# RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS, RPN_ANCHOR_STRIDE