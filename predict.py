"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import tensorflow

if hasattr(tensorflow.compat, 'v1'):
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
else:
    tf = tensorflow

import numpy as np
import cv2
import os
import argparse
from matplotlib import pyplot as plt
from pyaxidraw import axidraw

parser = argparse.ArgumentParser(description='ShadeSketch')

parser.add_argument('--direction', type=str, default='810', help='light direction (suggest to choose 810, 210, 710)')
args = parser.parse_args()


def cond_to_pos(cond):
    cond_pos_rel = {
        '002': [0, 0, -1],
        '110': [0, 1, -1], '210': [1, 1, -1], '310': [1, 0, -1], '410': [1, -1, -1], '510': [0, -1, -1],
        '610': [-1, -1, -1], '710': [-1, 0, -1], '810': [-1, 1, -1],
        '120': [0, 1, 0], '220': [1, 1, 0], '320': [1, 0, 0], '420': [1, -1, 0], '520': [0, -1, 0], '620': [-1, -1, 0],
        '720': [-1, 0, 0], '820': [-1, 1, 0],
        '130': [0, 1, 1], '230': [1, 1, 1], '330': [1, 0, 1], '430': [1, -1, 1], '530': [0, -1, 1], '630': [-1, -1, 1],
        '730': [-1, 0, 1], '830': [-1, 1, 1],
        '001': [0, 0, 1]
    }
    return cond_pos_rel[cond]


def axi_draw_svg(filename, ad):
    ad.plot_setup(filename)
    ad.plot_run()

def normalize_cond(cond_str):
    _cond_str = cond_str.strip()

    if len(_cond_str) == 3:
        return cond_to_pos(_cond_str)

    if ',' in _cond_str:
        raw_cond = _cond_str.replace('[', '').replace(']', '').split(',')
        if len(raw_cond) == 3:
            return raw_cond

    return [-1, 1, -1]


def predict():
    ad = axidraw.AxiDraw()
    output_dir = './output'
    input_dir = './input_images'
    image_size = 320
    use_smooth = False
    use_norm = False 
    threshold = 200
    direction = args.direction
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cam.read()
    print("Shape of Cam Grab: ")
    print(frame.shape)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load models
    with tf.gfile.FastGFile('./models/linesmoother.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineSmoother')

    with tf.gfile.FastGFile('./models/linenorm.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineNorm')

    with tf.gfile.FastGFile('./models/lineshader.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineShader')

    # Run through folders
    with tf.Session() as sess:
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                line_path = os.path.join(root, name)
                print('Running inference for %s ...' % line_path)

                img_temp = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
                
                thresh = 160   
                img_not_cropped = cv2.threshold(img_temp, thresh, 255, cv2.THRESH_BINARY)[1]
                blurred = cv2.blur(img_not_cropped, (3,3))
                canny = cv2.Canny(blurred, 50, 200)
                ## find the non-zero min-max coords of canny
                pts = np.argwhere(canny>0)
                y1,x1 = pts.min(axis=0)
                y2,x2 = pts.max(axis=0)

                ## crop the region
                img = img_not_cropped[y1:y2, x1:x2]
                #comp = 0.8 * img + 0.2 * shade

                cv2.imwrite(os.path.join('./data', 'regular.png'), img)
                #os.system("cd data; convert regular.png pnm_shading_regular.pnm; potrace pnm_shading_regular.pnm -s -o regular_svg.svg")

                #axi_draw_svg("./data/regular_svg.svg", ad)

                #os.remove("./data/regular.png")
                # Resize image
                s = image_size
                h, w = img.shape[:2]

                imgrs = cv2.resize(img, (s, s))

                # Threshold image
                if threshold > 0:
                    _, imgrs = cv2.threshold(imgrs, threshold, 255, cv2.THRESH_BINARY)

                # Prepare for inference
                tensors = np.reshape(imgrs, (1, s, s, 1)).astype(np.float32) / 255.
                ctensors = np.expand_dims(normalize_cond(direction), 0)

                # Run inference
                if use_smooth or threshold > 0:
                    tensors = sess.run(
                        'lineSmoother/conv2d_9/Sigmoid:0',
                        {
                            'lineSmoother/input_1:0': tensors
                        }
                    )
                    smoothResult = tensors

                if use_norm:
                    tensors = sess.run(
                        'lineNorm/conv2d_9/Sigmoid:0',
                        {
                            'lineNorm/input_1:0': tensors
                        }
                    )
                    normResult = tensors

                tensors = sess.run(
                    'lineShader/conv2d_139/Tanh:0',
                    {
                        'lineShader/input_1:0': ctensors,
                        'lineShader/input_2:0': 1. - tensors
                    }
                )
                shadeResult = tensors

                # Save result
                shade = (1 - (np.squeeze(shadeResult) + 1) / 2) * 255.
                shade = cv2.resize(shade, (w, h))
                """ print(shade.shape)
                imgplot = plt.imshow(shade, cmap='gray')
                plt.show() """
                path = 'C:/Users/Lenovo/Documents/RobotShading/ShadeSketch/data'
                cv2.imwrite("shading.png",shade)
                os.system("magick shading.png pnm_shading.pnm")
                os.system("potrace pnm_shading.pnm -s --pagesize A4 -r 96 -o final.svg")
                
                os.system("inkscape --batch-process --actions=\"de.vektorrascheln.hatch.noprefs; export-filename:hatchtest.svg; export-width:793; export-height:1123; export-dpi:96; export-do\" final.svg")
                
                os.remove("shading.png")
                os.remove("pnm_shading.pnm")
                os.remove("final.svg")
                #img = cv2.imread("test.png")

                
                #cv2.imwrite(os.path.join('./output', name), comp)


if __name__ == '__main__':
    predict()
