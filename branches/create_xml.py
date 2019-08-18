from lxml import etree
from glob import glob
import numpy as np
import tensorflow as tf
from PIL import Image
# from core import utils
import cv2

def read_coco_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]

    return input_tensor, output_tensors


def py_nms(boxes, scores,iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes => shape of [-1, 4], the value of '-1' means that dont know the
                        exact number of boxes
               scores => shape of [-1,]
               max_boxes => representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh => representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on cpu ---------------------------------------*/
    Arguments:
        boxes ==> shape [1, 10647, 4]
        scores ==> shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def creat_object_xml(_folder,_filename,_path,_width,_height,_name,_bndbox):
    # 固定参数
    _source={'database':'Unknown'}
    _size={'width':_width,'height':_height,'depth':'3'}
    _segmented='0'
    # 参数字典
    # _bndbox = {'xmin':_xmin,'ymin':_ymin,'xmax':_xmax,'ymax':_ymax}
    _object={'name':_name,'pose':'Unspecified','truncated':'0','difficult':'0','bndbox':_bndbox}
    head = {'folder': _folder, 'filename':_filename,
            'path':_path,'source':_source,'size':_size,'segmented':_segmented,'object':_object}
    # 创建xml
    root = etree.Element('annotation')
    for k, v in head.items():
        if not isinstance(v,dict):
            child = etree.Element(k)
            child.text = v
            root.append(child)
        elif k == 'object':
            for i in range(len(v['name'])):
                child = etree.Element(k)
                child1 = etree.Element('name')
                child1.text = str(int(v['name'][i])).replace('10','finger')
                child2 = etree.Element('pose')
                child2.text = v['pose']
                child3 = etree.Element('truncated')
                child3.text = v['truncated']
                child4 = etree.Element('difficult')
                child4.text = v['difficult']
                child5 = etree.Element('bndbox')
                # bbox 写入
                child5_1 = etree.Element('xmin')
                child5_1.text = str(int(round(v['bndbox'][i][0])))
                child5_2 = etree.Element('ymin')
                child5_2.text = str(int(round(v['bndbox'][i][1])))
                child5_3 = etree.Element('xmax')
                child5_3.text = str(int(round(v['bndbox'][i][2])))
                child5_4 = etree.Element('ymax')
                child5_4.text = str(int(round(v['bndbox'][i][3])))
                child5.append(child5_1)
                child5.append(child5_2)
                child5.append(child5_3)
                child5.append(child5_4)
                # 全部写入
                child.append(child1)
                child.append(child2)
                child.append(child3)
                child.append(child4)
                child.append(child5)
                root.append(child)
        else:
            child = etree.Element(k)
            for k1, v1 in v.items():
                if not isinstance(v1, dict):
                    child1 = etree.Element(k1)
                    child1.text = v1
                    child.append(child1)
                else:
                    child1 = etree.Element(k1)
                    for k2, v2 in v1.items():
                        child2 = etree.Element(k2)
                        child2.text = v2
                        child1.append(child2)
                    child.append(child1)
            root.append(child)
    tree = etree.ElementTree(root)
    tree.write(_path.replace('jpg','xml'), pretty_print=True)


def yolo_detect_to_xml(input_h,input_w,classes_names_path,images_dir,pb_model_path):
    classes = read_coco_names(classes_names_path)
    num_classes = len(classes)
    cpu_nms_graph = tf.Graph()
    input_tensor, output_tensors = read_pb_return_tensors(cpu_nms_graph, pb_model_path,
                                                                ["Placeholder:0", "concat_9:0", "mul_6:0"])
    with tf.Session(graph=cpu_nms_graph) as sess:
        for i_path in glob(images_dir+'*.jpg'):
            try:
                img = Image.open(i_path)
                img_resized = np.array(img.resize(size=(input_w, input_h)), dtype=np.float32)
                img_resized = img_resized / 255.
                boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})  # (1, 10647, 4) (1, 10647, 11)
                boxes, scores, labels = cpu_nms(boxes, scores, num_classes, score_thresh=0.3, iou_thresh=0.5)  # (10,4) (10,) (10,)
                # save xml
                _folder = i_path.split('/')[-2]
                _filename = i_path.split('/')[-1]
                _width, _height = img.size
                ratio_w, ratio_h = _width/input_w, _height/input_h
                boxes[:, [0, 2]] *= ratio_w
                boxes[:, [1, 3]] *= ratio_h
                _path = i_path
                creat_object_xml(_folder,_filename,_path,str(_width),str(_height),labels,boxes)
                # show
                # image = utils.draw_boxes(img, boxes, scores, labels, classes, [input_h, input_w], show=True)
            except TypeError as e:
                print(e)

def ssd_detect_to_xml(input_h,input_w,images_dir,pb_model_path):
    cpu_nms_graph = tf.Graph()
    input_tensor, output_tensors = read_pb_return_tensors(
        cpu_nms_graph, pb_model_path,
        ["image_tensor:0", "detection_boxes:0", "detection_scores:0",'detection_classes:0','num_detections:0'])
    _boxes, _scores, _classes, _num_detections = output_tensors
    with tf.Session(graph=cpu_nms_graph) as sess:
        for i_path in glob(images_dir+'*.jpg'):
            try:
                image = cv2.imread(i_path)
                image_np_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num_detections) = sess.run([_boxes, _scores, _classes, _num_detections],
                                                                       feed_dict={input_tensor: image_np_expanded})  # (1,100,4) (1,100) (1,100)

                boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
                boxes = boxes[scores != 0]*input_w
                classes = classes[scores != 0]-1
                scores = scores[scores!=0]
                indices = py_nms(boxes, scores, 0.1)
                boxes,classes = boxes[indices],classes[indices]

                # save xml
                _folder = i_path.split('/')[-2]
                _filename = i_path.split('/')[-1]
                _height, _width, _ = image.shape
                ratio_w, ratio_h = _width/input_w, _height/input_h
                boxes[:, [0, 2]] *= ratio_h
                boxes[:, [1, 3]] *= ratio_w
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
                _path = i_path
                creat_object_xml(_folder,_filename,_path,str(_width),str(_height),classes,boxes)
            except TypeError as e:
                print(e)


if __name__ == '__main__':
    # # yolo
    # input_h, input_w = 416, 416
    # classes_names_path = './data/number.names'
    # images_dir = "/home/gcl/number_research/data/VOCdevkit/VOC2012/test_data/"  # 181,
    # pb_model_path = "./yolov3_gpu_nms.pb"
    # yolo_detect_to_xml(input_h,input_w,classes_names_path,images_dir,pb_model_path)
    
    # ssd
    input_h, input_w = 300, 300
    images_dir = "/home/gcl/number/"  # 181,
    pb_model_path = "/home/gcl/work/frozen_inference_graph.pb"
    ssd_detect_to_xml(input_h, input_w, images_dir, pb_model_path)
