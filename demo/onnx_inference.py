import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import warnings
from imageio import imread
import pandas as pd
import time
import json
import urllib.request
from onnx import numpy_helper
import onnx
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime

session = onnxruntime.InferenceSession("/home/vansin/1.onnx")
print(session)


warnings.filterwarnings('ignore')
# display images in notebook

ssd_onnx_model = "/home/vansin/1.onnx"
tiny_yolov3_onnx_model = r"E:\BraveDownloads\tiny-yolov3-11\yolov3-tiny.onnx"

img_file = "./demo/demo.jpg"

# Preprocess and normalize the image


def preprocess(img_file, w, h):
    input_shape = (1, 3, w, h)
    img = Image.open(img_file)
    ori_size = img.size
    img = img.resize((w, h), Image.BILINEAR)
    # convert the input data into the float32 input
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (
            img_data[:, i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data.astype('float32'), np.array(img), ori_size

# %% SSD模型推理


def infer_ssd(onnx_model: str):
    # Run the model on the backend
    session = onnxruntime.InferenceSession(onnx_model, None)

    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    b, c, input_img_h, input_img_w = session.get_inputs()[0].shape

    # print(len(session.get_outputs()))
    print('Input Name:', input_name)
    print('Output Name:', output_name)
    # 符合https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd 模型的输入要求
    input_data, raw_img, ori_size = preprocess(
        img_file, input_img_w, input_img_h)
    print('输入图像大小：', input_data.shape)

    ori_w, ori_h = ori_size

    start = time.time()
    raw_result = session.run([], {input_name: input_data})
    end = time.time()
    print('推理时间：', end-start, 's')

    bboxes = raw_result[0][0]  # 200x4
    # labels = raw_result[1].T # 200x1
    # scores = raw_result[2].T # 200x1，结构已经按照得分从高到低的顺序排列

    # bboxes = np.array(bboxes)
    # bboxes = bboxes[bboxes[4]>0.6]

    fig, ax = plt.subplots(1)
    ax.imshow(raw_img)

    # LEN = np.sum(np.where(scores>0.6,1,0))

    for bbox in bboxes:

        if bbox[4] < 0.8:
            continue

        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', fill=False)

        ax.add_patch(rect)

    plt.show()

# infer_ssd(ssd_onnx_model)
# %% yolov3-tiny模型推理


def infer_tiny_yolov3(tiny_yolov3_onnx_model: str):

    # Run the model on the backend
    session = onnxruntime.InferenceSession(tiny_yolov3_onnx_model, None)

    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # print(len(session.get_outputs()))
    print('Input Name:', input_name)
    print('Output Name:', output_name)

    # 符合https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3 模型的输入要求
    input_data, raw_img = preprocess(img_file, 608, 608)
    print('输入数据大小：', input_data.shape)
    print('输入数据类型：', input_data.dtype)
    image_size = np.array(
        [raw_img.shape[1], raw_img.shape[0]], dtype=np.float32).reshape(1, 2)
    print('输入图像大小：', image_size.shape)
    start = time.time()
    raw_result = session.run([], {input_name: input_data,
                                  'image_shape': image_size})
    end = time.time()
    print('推理时间：', end-start, 's')
    yolonms_layer_1 = raw_result[0]
    yolonms_layer_1_1 = raw_result[1]
    yolonms_layer_1_2 = raw_result[2]
    print(yolonms_layer_1.shape, yolonms_layer_1_1.shape, yolonms_layer_1_2.shape)
    # print(yolonms_layer_1_2)


infer_ssd(ssd_onnx_model)
