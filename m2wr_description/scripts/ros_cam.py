#!/usr/bin/env python
import os
import sys

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

import cv2
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(FILE.parents[0].as_posix())

# from models.experimental import attempt_load
from utils.downloads import attempt_download
# from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
# from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
#     apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (set_logging, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort






class Camera_subscriber:

    def __init__(self):
        self.bridge = CvBridge()
        config_deepsort = ROOT / 'deep_sort_pytorch/configs/deep_sort.yaml'
        deep_sort_weights = ROOT / 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        weights = 'yolov5s.pt'  # model.pt path(s)
        data = ROOT / 'data/coco128.yaml'
        self.imgsz = (640, 640)  # inference size (pixels)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.stride = 32
        self.dnn = False
        device_num = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        update = False  # update all models
        name = 'exp'  # save results to project/name

        # Initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort  = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

        # Initialize
        set_logging()
        self.device = select_device(device_num)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.dnn, data=data, fp16=self.half)  # load model
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size


        # Second-stage classifier
        self.classify = False
        # if self.classify:
        #     self.modelc = load_classifier(name='resnet50', n=2)  # initialize
        #     self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(
        #         self.device).eval()

        # Dataloader
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        # if self.device.type != 'cpu':
        #     self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        self.model.warmup(imgsz=(1, 3, (640, 640))) # warmup

        self.subscription = rospy.Subscriber("m2wr/camera/image_raw", Image, self.callback)
        
    def callback(self, data):
        t0 = time.time()
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # check for common shapes
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=self.rect)[0] 
        # img = img[np.newaxis, :, :, :]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        # img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_sync()
        pred = self.model(img,
                          augment=self.augment,
                          visualize=increment_path('features', mkdir=True) if self.visualize else False)
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        t2 = time_sync()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, img0)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])  # xywh of object
                confs = det[:, 4]  # cof of object
                clss = det[:, 5]  # class of object

                # for *xyxy, conf, cls in reversed(det):
                #     c = int(cls)  # integer class
                #     label = None if self.hide_labels else (
                #         self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                #     annotator.box_label(xyxy, label, color=colors(c, True))

                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img0)
                # drax boxes
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {self.names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        # centure = realsence_show(img0, 2, bboxes, label, color=colors(c, True))
            
            img0 = annotator.result()
            cv2.imshow("IMAGE", img0)
            cv2.waitKey(3)
        print(s) 

def main(args):
    yolo = Camera_subscriber()
    rospy.init_node("camera_yolo_ros", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutdow')
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
