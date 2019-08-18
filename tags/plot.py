from glob import glob
import os
import cv2
import numpy as np
import argparse
import sys
import time
from extract_num_read import num_read_main
# Set up camera constants
IM_WIDTH = 640
IM_HEIGHT = 640
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

NUM_CLASSES = 11
# Grab path to current working directory
CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object_detection_number.pbtxt')
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
### Picamera ###
def path_to_id(img_xml_path):
    f_paths = glob(img_xml_path+'*.jpg')
    return [int(i.split('/')[-1].rstrip('.jpg')) for i in f_paths]

import xml.etree.ElementTree as ET
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
    return boxes,classes
ids = sorted(path_to_id('img_input/'))
cv2.namedWindow('Object detector',0)
cv2.resizeWindow('Object detector',IM_WIDTH,IM_HEIGHT)
for i in ids:
    i = 'img_input/'+str(i)
    t1 = cv2.getTickCount()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = cv2.imread(i+'.jpg')
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    boxes,classes =load_pascal_annotation(i+'.xml')
    # print(classes)
    read_number = num_read_main(bboxs=boxes[1:],finger_bbox=np.array([boxes[0]]),category=np.array(classes[1:]))
    # print(read_number)
    scores = len(boxes)*[0.9]
    # Draw the results of the detection (aka 'visulaize the results')
    for i in range(len(boxes)):
        c1 = tuple(boxes[i][:2].astype(int))
        c2 = tuple(boxes[i][2:].astype(int))
        cv2.rectangle(frame, c1,c2,color=[0,0,255],thickness=3)  # filled
        cv2.putText(frame, classes[i],(c1[0]+6,c1[1]-6),font,1,(0,0,255),2,cv2.LINE_AA)
    x,y = (boxes[0][0]/2+boxes[0][2]/2).astype(int),boxes[0][-1].astype(int)+5
    cv2.line(frame,(x,y),(x,y+100),color=[255,255,0],thickness=3)
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,"read_number: {0:.4f}cm".format(read_number),(200,50),font,1,(0,0,255),2,cv2.LINE_AA)
    time.sleep(1.5)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq

    frame_rate_calc = 1/time1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
