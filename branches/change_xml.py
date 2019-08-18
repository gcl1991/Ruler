import xml.etree.ElementTree as ET
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob


def change_wh_xml_single(xml_fp, new_w=1024):
    tree = ET.parse(xml_fp)
    root = tree.getroot()
    w, h = root.find('size').find('width').text,root.find('size').find('height').text
    ratio = new_w/int(w)
    new_h = int(int(h)*ratio)
    root.find('size').find('width').text = str(new_w)
    root.find('size').find('height').text = str(new_h)
    for object_node in root.iter('object'):
        bbox_x0 = object_node.find('bndbox').find('xmin').text
        bbox_y0 = object_node.find('bndbox').find('ymin').text
        bbox_x1 = object_node.find('bndbox').find('xmax').text
        bbox_y1 = object_node.find('bndbox').find('ymax').text
        object_node.find('bndbox').find('xmin').text = str(int(int(bbox_x0) * ratio))
        object_node.find('bndbox').find('ymin').text = str(int(int(bbox_y0) * ratio))
        object_node.find('bndbox').find('xmax').text = str(int(int(bbox_x1) * ratio))
        object_node.find('bndbox').find('ymax').text = str(int(int(bbox_y1) * ratio))
    tree.write(xml_fp)



def change_wh_jpg_single(jpg_fp,new_w=1024):
    im = cv2.imread(jpg_fp)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    w, h = im.size
    new_h = int(h*new_w/w)
    im = im.resize((new_w, new_h), Image.BILINEAR)
    im.save(jpg_fp)


for i in tqdm(glob('/home/gcl/repeat_number/'+'*.xml')):
    change_wh_xml_single(i)

for i in tqdm(glob('/home/gcl/repeat_number/'+'*.jpg')):
    change_wh_jpg_single(i)