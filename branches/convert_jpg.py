from glob import glob
from PIL import Image
import matplotlib.pylab as plt
import cv2
from tqdm import tqdm
img_w = 512
for i in tqdm(glob('/home/gcl/number/'+'*.jpg')):
    im = cv2.imread(i)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    #im = im.transpose(Image.ROTATE_180)
    w,h = im.size
    img_h = int(h*img_w/w)
    im = im.resize((img_w,img_h),Image.BILINEAR)
    im.save(i)
