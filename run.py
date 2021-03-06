'''
Original File: run_webcam.py
Project: MobilePose-PyTorch
File Created: Monday, 11th March 2019 12:47:30 am
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 2nd Februrary 2021 1:00:00 pm
Modified By: Daniel Rodriguez (drod11375@knights.ucf.edu>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''
import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
from torchvision import models

from network_modules import DUC, MobileNetV2
from network import CoordRegressionNetwork
from estimator import ResEstimator

def crop_camera(image, ratio=0.15):
    height = image.shape[0]
    width = image.shape[1]
    mid_width = width / 2.0
    width_20 = width * ratio
    crop_img = image[0:int(height), int(mid_width - width_20):int(mid_width + width_20)]
    return crop_img

if __name__ == '__main__':

    # load the model
    model_path = './models/mobilenetv2_224_adam_best.t7'

    net = CoordRegressionNetwork(
        n_locations=16,
        backbone='mobilenetv2'
    ).to('cpu')

    estimator = ResEstimator(
        model_path,
        net,
        224
    )

    # initialize the camera
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    image = crop_camera(image)

    while True:
        
        # get frame
        ret_val, image = cam.read()
        image = crop_camera(image)

        # infer through network
        humans = estimator.inference(image)
        # image = ResEstimator.draw_humans(
        #     image,
        #     humans,
        #     imgcopy=False
        # )

        draw_rects = True

        # Head
        # 8 -> 9
        # pt1=(humans[9][0]-75,humans[9][1])
        # pt2=(humans[8][0]+75,humans[8][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if (draw_rects):
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w, y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
            # cv2.imshow('8->9', image[y:y+h, x:x+w])

        # 12 -> 11
        # R_should -> R_elb
        # Switch points to mirror body parts
        # pt1=(humans[12][0]-75,humans[12][1]-75)
        # pt2=(humans[11][0]+50,humans[11][1]+50)
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w,y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x-h].size
        # if check > 0:
        #     cv2.imshow('12->11', image[y:y+h, x:x-h])
        
        # 11 -> 10
        # R_elb -> R_wrist
        # pt2=(humans[10][0],humans[10][1])
        # pt1=(humans[11][0],humans[11][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w,y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
        #     cv2.imshow('11->10', image[y:y+h, x:x+w])

        # 2 -> 1
        # R-hip -> r-knee
        # pt1=(humans[2][0]+50,humans[2][1])
        # pt2=(humans[1][0]-50,humans[1][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w,y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x+w:x].size
        # if check > 0:
        #     cv2.imshow('2->1', image[y:y+h, x+w:x])

        # 1 -> 0
        # R-knee -> r_ankl
        # pt1=(humans[1][0]+50,humans[1][1])
        # pt2=(humans[0][0]-50,humans[0][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w,y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x+w:x].size
        # if check > 0:
        #     cv2.imshow('1->0', image[y:y+h, x+w:x])


        # 13 -> 14
        # L_should -> L_elb
        # pt1=(humans[13][0]-25,humans[13][1]-25)
        # pt2=(humans[14][0]+25,humans[14][1]+25)
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w, y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
        #     cv2.imshow('13->14', image[y:y+h, x:x+w])

        # 14 -> 15
        # L_elb -> L_wrist
        # pt1=(humans[14][0]-50,humans[14][1]-50)
        # pt2=(humans[15][0]+75,humans[15][1]+75)
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w, y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
        #     cv2.imshow('14->15', image[y:y+h, x:x+w])

        # 3 -> 4
        # L_hip -> L_knee
        # pt1=(humans[3][0]-50,humans[3][1])
        # pt2=(humans[4][0]+50,humans[4][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w, y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
        #     cv2.imshow('3->4', image[y:y+h, x:x+w])

        # 4 -> 5
        # l_knee -> l_ank
        # pt1=(humans[4][0]-50,humans[4][1])
        # pt2=(humans[5][0]+50,humans[5][1])
        # x = pt1[0]
        # y = pt1[1]
        # w = pt2[0] - pt1[0]
        # h = pt2[1] - pt1[1]
        # if draw_rects:
        #     cv2.rectangle(
        #         image,
        #         pt1=(x,y),
        #         pt2=(x+w, y+h),
        #         color=(0,0,255),
        #         thickness=2
        #     )
        # check = image[y:y+h, x:x+w].size
        # if check > 0:
        #     cv2.imshow('4->5', image[y:y+h, x:x+w])


        cv2.imshow('MobileNetV2+DUC+DSNT', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
