import numpy
import random
import sys
from PIL import Image

from goldfish import GOLDFISH_RNG_SEED

'''
Embedder class that uses LSB to hide a message in an image
  In the case of RaaS, the message will be a key to a database
  entry for a rendered image
'''
class LSBEmbedder(object):
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(GOLDFISH_RNG_SEED)

    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        locations = zip(self.rng.sample(xrange(height), len(bin_message)),
                        self.rng.sample(xrange(width ), len(bin_message)))
        channels = [self.rng.choice(range(len(bands)))
                    for i in range(len(bin_message))]

        for b_i, bit in enumerate(bin_message):
            original = format(bands[channels[b_i]][locations[b_i]], 'b')[:-1]
            bands[channels[b_i]][locations[b_i]] = int(original+bit, 2)

        return Image.merge('RGB', [Image.fromarray(band) for band in bands])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output

class HistogramEmbedder(object):
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(GOLDFISH_RNG_SEED)

    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        # get the histogram and alter the image channels
        hists = [Image.fromarray(b).histogram() for b in bands]
        peaks = [self._get_max_point(hist) for hist in hists]
        zeros = [self._get_min_point(hist) for hist in hists]
        print peaks
        print zeros
        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        channels = [self.rng.choice(range(len(bands)))
                for i in range(len(bin_message))]
        # we can shift values to the right and embed the data at the same time
        current_index = 0
        done_embedding = False
        # for each band, shift right and embed data if we are not done and if
        # this is the right band for the current bit
        for b_i in range(len(bands)):
            for i in range(width):
                for j in range(height):
                    if peaks[b_i] <= bands[b_i][i,j] <= zeros[b_i]:
                        bands[b_i][i,j] += 1
                    elif bands[b_i][i,j] == peaks[b_i]-1 and not done_embedding:
                        if current_index >= len(bin_message):
                            done = True
                        elif b_i == channels[current_index]:
                            bands[b_i] += int(bin_message[current_index])
                            current_index += 1

        return Image.merge('RGB', [Image.fromarray(band) for band in bands])


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

