from PIL import Image

def reorient_image(im):
    try:
        image_exif = im._getexif()
        image_orientation = image_exif[274]
        if image_orientation in (2,'2'):
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in (3,'3'):
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in (4,'4'):
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (5,'5'):
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (6,'6'):
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in (7,'7'):
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (8,'8'):
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im
from glob import glob
from tqdm import tqdm
imags_dir = '/media/gcl/Ubuntu-Serv/number_2019_5_12/'

imags_file = glob(imags_dir+'*.JPG')
for i in tqdm(imags_file):
    im = Image.open(i)
    im = reorient_image(im)
    im.save(i)