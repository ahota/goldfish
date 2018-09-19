import sys
sys.path.append('../')
from goldfish.watermarker import Watermarker

length = 413

wm = Watermarker(chan='R', bits_per_pixel=4)
message = wm.extract('../images/teapot-watermark.png',
        message_length=length*8)
print message
