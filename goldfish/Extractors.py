import numpy
import random
import sys
from PIL import Image

from goldfish import GOLDFISH_RNG_SEED

'''
Extractor class that uses extracts a message from the LSB of image pixels
'''
class LSBExtractor(object):
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(GOLDFISH_RNG_SEED)

    def extract(self, image, message_length=256):
        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        locations = zip(self.rng.sample(xrange(height), message_length),
                        self.rng.sample(xrange(width ), message_length))
        channels = [self.rng.choice(range(len(bands)))
                    for i in range(message_length)]

        bin_message = ''.join(
                [format(bands[channels[b_i]][locations[b_i]],'b')[-1]
                 for b_i in range(message_length)])

        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output


