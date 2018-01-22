import numpy
import random
from PIL import Image
import sys

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
    LSB method uses OPAP from C. Chan and L. Cheng, Hiding data in images
    by simple LSB substitution, 2004.
    '''
    def __init__(self, bits_per_pixel=2, chan='R', debug=False):
        self.rng = random.Random()
        self.debug = debug
        self.k_bits = bits_per_pixel
        self.symbols = []
        self.mode = 'RGB'
        self.band = self.mode.index(chan)
        # flag to know if this object did the embedding when extracting a
        # message from an image. This prevents the extractor from doing
        # error-checking against the symbols list, which will be empty if
        # this object did not do the embedding
        self.did_embed = False

    @seeded
    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        self.did_embed = True
        width, height = image.size

        yimage = image.convert(self.mode)
        if len(image.getbands()) > 3:
            bands = self._get_bands(yimage)[:3]
        else:
            bands = self._get_bands(yimage)

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])

        n_places = len(bin_message)/self.k_bits
        self._debug_message('n_places before', n_places)
        if len(bin_message) % self.k_bits != 0:
            n_places += 1
        self._debug_message('n_places after', n_places)

        # locations to embed data in, randomly located across the image
        locations = list(numpy.ndindex((height, width)))
        self.rng.shuffle(locations)
        self._debug_message('locations[:10]', locations[:10])
        #locations = zip(self.rng.sample(xrange(height), n_places),
        #                self.rng.sample(xrange(width ), n_places))
        luma = bands[self.band]

        for b_i in range(n_places):
            # given a symbol and a location, embed the symbol in the last
            # k bits of the pixel value at the location
            symbol = bin_message[b_i*self.k_bits:b_i*self.k_bits+self.k_bits]
            original_val = luma[locations[b_i]]
            original = format(original_val, 'b').zfill(8)
            self.symbols.append([symbol])
            stego = original[:-self.k_bits] + symbol
            stego_val = int(stego, 2)
            # apply OPAP on the stego pixel
            # this is supposed to minimize the error introduced in embedding
            # first find which interval the error exists within
            delta = stego_val - original_val
            optimized = stego_val # default for some cases
            # three intervals
            if 2**(self.k_bits-1) < delta < 2**self.k_bits:
                if stego_val >= 2**self.k_bits:
                    optimized = stego_val - 2**self.k_bits
            elif -2**(self.k_bits-1) <= delta <= 2**(self.k_bits-1):
                pass # just use the default option above
            elif -2**(self.k_bits) < delta < -2**(self.k_bits-1):
                if stego_val < 256 - 2**(self.k_bits):
                    optimized = stego_val + 2**self.k_bits
            luma[locations[b_i]] = optimized

        bands[self.band] = luma

        watermarked = Image.merge(self.mode,
                [Image.fromarray(band) for band in bands])
        return watermarked

    @seeded
    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size

        yimage = image.convert(self.mode)
        bands = self._get_bands(yimage)

        n_places = message_length/self.k_bits
        leftovers = message_length % self.k_bits
        self._debug_message('n_places before', n_places)
        if leftovers != 0:
            n_places += 1
            self._debug_message('n_places after', n_places)
        self._debug_message('leftovers', leftovers)

        locations = list(numpy.ndindex((height, width)))
        self.rng.shuffle(locations)
        self._debug_message('locations[:10]', locations[:10])
        #locations = zip(self.rng.sample(xrange(height), n_places),
        #                self.rng.sample(xrange(width ), n_places))
        luma = bands[self.band]

        bin_message = ''
        for b_i in range(n_places):
            data = format(luma[locations[b_i]], 'b').zfill(8)
            if b_i == n_places - 1 and leftovers != 0:
                symbol = data[-leftovers:]
            else:
                symbol = data[-self.k_bits:]
            if self.debug and self.did_embed:
                self.symbols[b_i].append(symbol)
            bin_message += symbol

        if self.debug and self.did_embed:
            for symbol in self.symbols:
                print symbol[0], '->', symbol[1]

        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output

    def _debug_message(self, *args):
        if not self.debug:
            return
        print 'DEBUG:',
        for arg in args:
            print arg,
        print
