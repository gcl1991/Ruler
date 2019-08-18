from glob import glob
import os
import cv2
import numpy as np
import argparse
import sys
import time
import logging
from extract_num_read import num_read_main
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

logging.basicConfig(filename='log',level=logging.DEBUG,filemode='w')


def path_to_id(img_xml_path):
    return glob(img_xml_path+'*.xml')


def load_pascal_annotation(filename):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.float64)
    classes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        cls = obj.find('name').text.lower().strip()
        classes.append(cls)
        boxes[ix, :] = [x1, y1, x2, y2]
    return boxes,np.array(classes)

def plt_bbox_on_im(image,boxes,classes,finger_bbox):
    'cv2 在图片绘制bbox和指针'
    thickness = 2
    finger_size = 150
    fontscale = 2
    org_bottom_left_gap = 6
    for i in range(len(boxes)):
        c1 = tuple(boxes[i][:2].astype(int))
        c2 = tuple(boxes[i][2:].astype(int))
        cv2.rectangle(image, c1, c2, color=[0, 0, 255], thickness=thickness)  # filled
        cv2.putText(image, str(int(classes[i])), (c1[0]+org_bottom_left_gap, c1[1]-org_bottom_left_gap), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 0, 255), thickness, cv2.LINE_AA)
    if finger_bbox.size:
        x, y = (finger_bbox[0] / 2 + finger_bbox[2] / 2).astype(int), (finger_bbox[1]).astype(int)
        logging.debug('x {} y {}'.format(x, y))
        cv2.line(image, (x, y), (x, y + finger_size), color=[255, 255, 0], thickness=thickness)

def main():
    input_path = '/home/gcl/repeat_number/'
    ids = sorted(path_to_id(input_path))
    for i in ids:
        try:
            frame = plt.imread(i.replace('xml','jpg'))
            logging.info('读取文件：\t{}'.format(i))
            boxes, classes = load_pascal_annotation(i)
            logging.debug('原始 boxes {} , classes {}'.format(boxes, classes))
            finger_index = np.argwhere(classes == 'finger').squeeze()
            logging.debug(finger_index)
            finger_bbox = boxes[finger_index]
            logging.debug(finger_bbox)
            boxes, classes = np.delete(boxes, finger_index, axis=0), np.delete(classes, finger_index, axis=0)
            logging.debug('删除指针 boxes {} , classes {}'.format(boxes, classes))
            logging.debug(boxes.size)
            if classes.size <= 1 or finger_index.size == 0:
                read_number = None
            else:
                read_number = num_read_main(bboxs=boxes, finger_bbox=finger_bbox,category=classes)
            print('读数：{}'.format(read_number))
            plt_bbox_on_im(frame,boxes,classes,finger_bbox)
            # import matplotlib
            # print(matplotlib.get_backend())
            fig = plt.figure()
            fig.canvas.manager.window.move(512,0)
            plt.imshow(frame)
            plt.title(i)
            plt.xlabel(str(read_number))
            plt.show()
        except KeyboardInterrupt:
            print('下一张')
if __name__ == '__main__':
    main()