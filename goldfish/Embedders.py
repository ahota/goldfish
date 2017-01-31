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

