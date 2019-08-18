from multiprocessing import Process
from luma.core.interface.serial import i2c, spi
from luma.core.render import canvas
from luma.oled.device import ssd1306, ssd1309, ssd1325, ssd1331, sh1106
from luma.core.threadpool import threadpool
import time
import threading

def print_result(indent,result):
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial)
    device.clear()
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        x = device.bounding_box[2]
        y = device.bounding_box[3]
        draw.text((int(x/2)-indent, int(y/2)), result, fill="white")


