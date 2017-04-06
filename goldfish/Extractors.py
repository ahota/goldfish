import numpy
import random
import sys
from PIL import Image

from goldfish import GOLDFISH_RNG_SEED
from matplotlib import pyplot
from scipy.fftpack import dct, idct

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
        pyplot.plot(hists[0], 'r')
        pyplot.plot(hists[1], 'g')
        pyplot.plot(hists[2], 'b')
        pyplot.show()
        peaks = [self._get_max_point(hist) for hist in hists]
        zeros = [self._get_min_point(hist) for hist in hists]
        print peaks
        print zeros
        #channels = [self.rng.choice(range(len(bands)))
        #        for i in range(message_length)]
        channels = [1] * message_length
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
        print bin_message[:32]
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
        if max_point == 0:
            max_point = 2
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

class EntropyExtractor(object):
    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(GOLDFISH_RNG_SEED)
        # used to get a list of DCT coefficients
        # from:
        # https://github.com/lot9s/lfv-compression/blob/master/scripts/
        # our_mpeg/zigzag.py
        self.zigzag = numpy.array(
                [[0,  1,  5,  6,  14, 15, 27, 28],
                 [2,  4,  7,  13, 16, 26, 29, 42],
                 [3,  8,  12, 17, 25, 30, 41, 43],
                 [9,  11, 18, 24, 31, 40, 44, 53],
                 [10, 19, 23, 32, 39, 45, 52, 54],
                 [20, 22, 33, 38, 46, 51, 55, 60],
                 [21, 34, 37, 47, 50, 56, 59, 61],
                 [35, 36, 48, 49, 57, 58, 62, 63]]
        )
        self.zigzagflatinverse = self.zigzag.flatten()
        self.zigzagflat = numpy.argsort(self.zigzagflatinverse)
        self.energy_threshold = 2000
        self.base_quantize_matrix = numpy.array(
                [[16,  11,  10,  16,  24,  40,  51,  61],
                 [12,  12,  14,  19,  26,  58,  60,  55],
                 [14,  13,  16,  24,  40,  57,  69,  56],
                 [14,  17,  22,  29,  51,  87,  80,  62],
                 [18,  22,  37,  56,  68, 109, 103,  77],
                 [24,  35,  55,  64,  81, 104, 113,  92],
                 [49,  64,  78,  87, 103, 121, 120, 101],
                 [72,  92,  95,  98, 112, 100, 103,  99]]
        )
        self.k = 33
        self.m = 5

    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        bin_message = ''
        #quantize_matrix = self._setup_quantize_matrix(quality)
        
        # just deal with the red band for now
        # split into quarter tiles
        # terrible variable names ahead, deal with it
        tw = width/2  # tile width
        th = height/2 # tile height
        psw = 8       # process-significant block width
        psh = 8       # process-significant block height
        tiles = numpy.array(
            [bands[0][i*tw:(i+1)*tw, j*th:(j+1)*th]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, tw, th)
        tile_messages = [[],[],[],[]]

        # embed the message in each tile
        bits_per_block = 4
        # for each tile
        for (ti, tj) in numpy.ndindex(2, 2):
            current_index = 0
            # divide the tile into 8x8 ps blocks
            ps_blocks = numpy.array(
                [tiles[ti, tj][i*psw:(i+1)*psw, j*psh:(j+1)*psh]
                for (i, j) in numpy.ndindex(tw/psw, th/psh)]
            ).reshape(tw/psw, th/psh, psw, psh)
            # for each block
            for (bi, bj) in numpy.ndindex(tw/psw, th/psh):
                if current_index >= message_length:
                    break
                # 2d dct
                block = dct(dct(ps_blocks[bi, bj].T, norm='ortho').T, norm='ortho')
                block = block.flatten()[self.zigzagflat] # get in zig zag order
                for bit_i in range(1, 1+bits_per_block):
                    first_bit = format(int(block[bit_i]), 'b').zfill(4)[-1]
                    third_bit = format(int(block[bit_i]), 'b').zfill(4)[-3]
                    #print format(int(block[bit_i]), 'b').zfill(3), first_bit, third_bit, block[bit_i]
                    if int(first_bit, 2) ^ int(third_bit, 2):
                        tile_messages[ti*2+tj].append(0)
                    else:
                        tile_messages[ti*2+tj].append(1)
                    current_index += 1

        # check the tile_messages we got
        summed = numpy.sum(tile_messages, axis=0)
        for s in summed:
            if s > 2:
                bin_message += '1'
            else:
                bin_message += '0'
        print bin_message[:64]

        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output
