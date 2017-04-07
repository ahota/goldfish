import numpy
import random
import sys
from PIL import Image
from scipy.fftpack import dct, idct

from goldfish import GOLDFISH_RNG_SEED
from matplotlib import pyplot

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
        pyplot.plot(hists[0], 'r')
        pyplot.plot(hists[1], 'g')
        pyplot.plot(hists[2], 'b')
        pyplot.show()
        peaks = [self._get_max_point(hist) for hist in hists]
        zeros = [self._get_min_point(hist) for hist in hists]
        print peaks
        print zeros
        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        print bin_message[:32]
        #channels = [self.rng.choice(range(len(bands)))
        #        for i in range(len(bin_message))]
        channels = [1] * len(bin_message)
        # we can shift values to the right and embed the data at the same time
        current_index = 0
        done_embedding = False
        # for each band, shift right and embed data if we are not done and if
        # this is the right band for the current bit
        for b_i in range(1, 2):
            for i in range(width):
                for j in range(height):
                    if peaks[b_i] <= bands[b_i][i,j] <= zeros[b_i]:
                        bands[b_i][i,j] += 1
                    elif bands[b_i][i,j] == peaks[b_i]-1 and not done_embedding:
                        if current_index >= len(bin_message):
                            done = True
                        elif b_i == channels[current_index]:
                            bands[b_i][i,j] += int(bin_message[current_index])
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
        if max_point == 0:
            max_point = 1
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

class EntropyEmbedder(object):
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
        self.energy_threshold = 500
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

    def embed(self, image, message, quality=75):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        quantize_matrix = self._setup_quantize_matrix(quality)
        
        # just deal with the blue band for now
        # split into quarter tiles
        # terrible variable names ahead, deal with it
        tw = width/2  # tile width
        th = height/2 # tile height
        psw = 8       # process-significant block width
        psh = 8       # process-significant block height
        tiles = numpy.array(
            [bands[2][i*tw:(i+1)*tw, j*th:(j+1)*th]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, tw, th)

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
                if current_index >= len(bin_message):
                    break
                # 2d dct
                block = dct(dct(ps_blocks[bi, bj].T, norm='ortho').T, norm='ortho')
                # check block energy
                orig_energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                if orig_energy < self.energy_threshold:
                    #ps_blocks[bi, bj][:] = 255
                    continue # skip this block
                # divide by the jpeg quantization matrix
                # image should resist up to <quality> jpeg compression
                block /= quantize_matrix
                block = block.flatten()[self.zigzagflat] # get in zig zag order
                # embed bits_per_block bits of the message into this ps block
                for bit_i in range(1, 1+bits_per_block):
                    if bin_message[current_index+bit_i-1] == '1':
                        # round coefficient to the nearest odd number
                        block[bit_i] = 2 * numpy.round((block[bit_i]+1)/2) - 1
                    else:
                        # round coefficient to the nearest even number
                        block[bit_i] = 2 * numpy.round(block[bit_i]/2)
                    #if ti == 1 and tj == 0 and shit < 10:
                    #    print block[bit_i]
                    '''
                    # this part comes from the Bo Li paper
                    third_bit = format(int(block[bit_i]), 'b').zfill(3)[-3]
                    original_val = block[bit_i]
                    if bin_message[current_index] == '1' and third_bit == '1':
                        block[bit_i] = 2 * int(numpy.round((block[bit_i]-1)/2.0)) + 1
                    elif bin_message[current_index] == '1' and third_bit == '0':
                        block[bit_i] = 2 * int(numpy.round(block[bit_i]/2))
                    elif bin_message[current_index] == '0' and third_bit == '1':
                        block[bit_i] = 2 * int(numpy.round(block[bit_i]/2))
                    else:
                        block[bit_i] = 2 * int(numpy.round((block[bit_i]-1)/2.0)) + 1
                    # set noticing parameter
                    d = block[self.k + bit_i]
                    d_prime = int(numpy.round(d))
                    d_bits = format(d_prime, 'b').zfill(5) # % (1 << 5), 'b')
                    # handle the negative sign python puts in
                    was_neg = False
                    if d_bits[0] == '-':
                        d_bits = d_bits[1:]
                        was_neg = True
                    if block[bit_i] == original_val:
                        # embed a 0 into the mth bit of the kth+bit_i coeff
                        d_np = d_bits[:-self.m]+'0'+d_bits[-self.m+1:]
                    else:
                        # embed a 1
                        d_np = d_bits[:-self.m]+'1'+d_bits[-self.m+1:]
                    # replace the negative if it was there
                    if was_neg:
                        d_np = int('-'+d_np, 2)
                    else:
                        d_np = int(d_np, 2)
                    block[self.k + bit_i] = d_np - d_prime + d
                    current_index += 1
                    '''
                # un-zigzag the block
                block = block[self.zigzagflatinverse].reshape(8, 8)
                # check the new energy
                new_energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                if new_energy < self.energy_threshold:
                    #ps_blocks[bi, bj][:] = 128
                    continue # skip this block
                else:
                    current_index += 4
                # multiply by quantization matrix
                block *= quantize_matrix
                # inverse dct
                block = idct(idct(block, norm='ortho').T, norm='ortho').T
                # reassign back to tile
                ps_blocks[bi, bj] = block
                '''
            if current_index < len(bin_message) - 1:
                print 'Tile', ti, tj, 'could only hold', current_index,
                print 'out of', len(bin_message), 'bits'
                '''
            # reassemble the tile
            tiles[ti, tj] = numpy.hstack([numpy.vstack(ps_blocks[:,i])
                for i in range(tw/psw)])

        # reassemble the tiles into a channel
        bands[2] = numpy.hstack([numpy.vstack(tiles[:, i]) for i in range(2)])

        return Image.merge('RGB', [Image.fromarray(b) for b in bands])

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output

    def _setup_quantize_matrix(self, quality):
        if quality < 50:
            s = 5000/quality
        else:
            s = 200 - 2*quality

        return numpy.floor((s*self.base_quantize_matrix + 50) / 100)

