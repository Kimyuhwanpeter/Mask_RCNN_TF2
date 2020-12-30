# -*- coding:utf-8 -*-
import tensorflow as tf

# https://github.com/ahmedfgad/Mask-RCNN-TF2/blob/master/mrcnn/model.py
RPN_ANCHOR_RATIOS = [0.5, 1, 2]

def idendity_block(input,
                   kernel_size,
                   filters,
                   use_bias=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    x = tf.keras.layers.Conv2D(filters=nb_filter1,
                               kernel_size=1,
                               strides=1,
                               use_bias=use_bias)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter2,
                               kernel_size=kernel_size,
                               strides=1,
                               padding="same",
                               use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter3,
                               kernel_size=1,
                               strides=1,
                               use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, input])
    x = tf.keras.layers.ReLU()(x)

    return x

def conv_block(input,
               kernel_size,
               filters,
               strides=2,
               use_bias=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    x = tf.keras.layers.Conv2D(filters=nb_filter1,
                               kernel_size=1,
                               strides=strides,
                               use_bias=use_bias)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter2,
                               kernel_size=kernel_size,
                               padding="same",
                               use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter3,
                               kernel_size=1,
                               padding="same",
                               use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = tf.keras.layers.Conv2D(filters=nb_filter3,
                                      kernel_size=1,
                                      strides=strides,
                                      use_bias=use_bias)(input)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x

def RPN(input, anchors_per_location, anchor_strides):

    shared = tf.keras.layers.Conv2D(filters=512,
                                    kernel_size=3,
                                    padding="same",
                                    activation="relu")(input)
    x = tf.keras.layers.Conv2D(2 * anchors_per_location,
                               kernel_size=1,
                               padding="valid",
                               activation="linear")(shared) # [B, h, w, anchors_per_location * 2]
    
    rpn_class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    rpn_probs = tf.keras.layers.Activation("softmax")(rpn_class_logits)

    x = tf.keras.layers.Conv2D(anchors_per_location * 4,
                               kernel_size=1,
                               padding="valid",
                               activation="linear")(shared) # [B, h, w, anchors_per_location * depth]
    # depth --> [x, y, log(w), log(h)]

    rpn_bbox = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

def ResNet_FPN(input_shape=(256, 256, 3), anchor_strides=1, architecture="resnet50", stage5=True):
    # resnet 101 or 50
    x = inputs = tf.keras.Input(input_shape)
    # https://arxiv.org/pdf/1703.06870.pdf

    # Resnet
    x = tf.keras.layers.ZeroPadding2D((3,3))(x)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    C1 = x = tf.keras.layers.MaxPooling2D(pool_size=(3,3),
                                          strides=(2,2),
                                          padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], strides=1)
    x = idendity_block(x, 3, [64, 64, 256])
    C2 = x = idendity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = idendity_block(x, 3, [128, 128, 512])
    x = idendity_block(x, 3, [128, 128, 512])
    C3 = x = idendity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = idendity_block(x, 3, [256, 256, 1024])
    C4 = x

    if stage5:
        x = conv_block(x, 3, [512, 512, 2048])
        x = idendity_block(x, 3, [512, 512, 2048])
        C5 = x = idendity_block(x, 3, [512, 512, 2048])
    else:
        C5 = None

    # FPN
    P5 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=1)(C5)
    P4 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P5),
                               tf.keras.layers.Conv2D(filters=256,
                                                      kernel_size=1)(C4)])
    P3 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P4),
                                tf.keras.layers.Conv2D(filters=256,
                                                       kernel_size=1)(C3)])
    P2 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P3),
                                tf.keras.layers.Conv2D(filters=256,
                                                       kernel_size=1)(C2)])
    
    P2 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                padding="same")(P2)
    P3 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                padding="same")(P3)
    P4 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                padding="same")(P4)
    P5 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                padding="same")(P5)
    P6 = tf.keras.layers.MaxPooling2D(pool_size=(1,1), strides=2)(P5)

    # output
    input_rpn_feature = [P2, P3, P4, P5, P6]
    input_mrcnn_feature = [P2, P3, P4, P5]

    # RPN
    layers_outputs = []
    for p in input_rpn_feature:
        layers_outputs.append(RPN(p, len(RPN_ANCHOR_RATIOS), anchor_strides=1))
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]

    outputs = list(zip(*layers_outputs))
    outputs = [tf.keras.layers.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs # 사실상 rpn_class와 rpn_bbox 두개 만 필요

    return tf.keras.Model(inputs=inputs, outputs=[rpn_class_logits, rpn_class, rpn_bbox])

# 레이어 테스트를 해보고 진행하자