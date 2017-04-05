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
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

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

class HistogramExtractor(object):
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(GOLDFISH_RNG_SEED)

    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image[:3])
        else:
            bands = self._get_bands(image)

        hists = [Image.fromarray(b).histogram() for b in bands]
        peaks = [self._get_max_point(hist) for hist in hists]
        zeros = [self._get_min_point(hist) for hist in hists]
        print peaks
        print zeros
        channels = [self.rng.choice(range(len(bands)))
                for i in range(message_length)]
        bin_message = ''

        current_index = 0
        done_extracting = False
        for i in range(width):
            if done_extracting:
                break
            for j in range(height):
                if done_extracting:
                    break
                chan = channels[current_index]
                if bands[chan][i,j] == peaks[chan] - 2:
                    bin_message += '0'
                    current_index += 1
                elif bands[chan][i,j] == peaks[chan] - 1:
                    bin_message += '1'
                    current_index += 1
                if current_index >= message_length:
                    done_extracting = True
        return ''.join([chr(int(bin_message[i:i+8], 2))
            for i in range(0, len(bin_message), 8)])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output

    def _get_max_point(self, hist):
        max_point = -1
        max_val = hist[-1]
        for i in range(len(hist)-1, -1, -1):
            if hist[i] > max_val:
                max_val = hist[i]
                max_point = i
        return max_point

    def _get_min_point(self, hist):
        min_point = -1
        min_val = hist[-1]
        for i in range(len(hist)-1, -1, -1):
            if hist[i] < min_val:
                min_val = hist[i]
                min_point = i
        if min_point == -1:
            min_point = 254
        return min_point

