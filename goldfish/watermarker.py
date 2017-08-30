import numpy
import random
from PIL import Image

def seeded(func):
    def func_wrapper(self, *args, **kwargs):
        self.rng.seed('suman')
        return func(self, *args, **kwargs)
    return func_wrapper

class Watermarker(object):
    '''
    Base watermarking algorithm class. Implements LSB embedding and
    extracting and exposes the embed() and extract() API for child
    classes
    '''
    def __init__(self):
        self.rng = random.Random()

    @seeded
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

    @seeded
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
