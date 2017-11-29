from watermarker import *
from scipy.fftpack import dct, idct
from unireedsolomon import rs
from bitstring import BitArray

import sys

class EnergyWatermarker(Watermarker):
    '''
    Embedder/extractor using Lee/Li's pre-inserted code method from
    J. Lee and B. Li. Self-Recognized Image Protection Technique that Resists
    Large-Scale Cropping. 2014.
    '''
    def __init__(self, bits=4, chan='luma', **kwargs):
        Watermarker.__init__(self, **kwargs)
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
        self.debug_vals = [] # used for checking embedded/extracted values
        self.chan = chan

    @seeded
    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size

        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])

        embed_band = ['luma', 'cb', 'cr'].index(self.chan)
        band = bands[embed_band]
        # using nomenclature from paper
        # - slices are quadrants of the image plane being watermarked
        # - process-significant (ps) blocks are subdivisions of the slices
        #   into which watermark data is embedded
        slice_w, slice_h = width/2, height/2
        block_w, block_h = max(8, slice_w/64), max(8, slice_h/64)

        # divide image plane into slices
        slices = numpy.array(
            [band[i*slice_w:(i+1)*slice_w, j*slice_h:(j+1)*slice_h]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, slice_w, slice_h)

        bits_per_block = 4

        did_print = False
        # embed the message in each slice
        # for each slice
        for (si, sj) in numpy.ndindex(2, 2):
            # index into the watermark data
            current_index = 0
            # divide current slice into ps blocks
            ps_blocks = numpy.array(
                [slices[si,sj][i*block_w:(i+1)*block_w, j*block_h:(j+1)*block_h]
                for (i, j) in numpy.ndindex(slice_w/block_w, slice_h/block_h)]
            ).reshape(slice_w/block_w, slice_h/block_h, block_w, block_h)

            # for each block
            for (bi, bj) in numpy.ndindex(slice_w/block_w, slice_h/block_h):
                if current_index >= len(bin_message):
                    break
                # 2D DCT
                block = dct(dct(ps_blocks[bi,bj].T,norm='ortho').T,norm='ortho')
                block = block.flatten()[self.zigzagflat] # get in zig zag order

                # embed <bits_per_block> bits of the message into this block
                for bit_i in range(1, 1+bits_per_block):
                    # this method is explained in a very complex way, but all
                    # it is doing is quantization with a delta of 2
                    # So, if you take the current coefficient B(j) and truncate
                    # it, the third LSB of it is b'
                    # if b' XOR the current watermark bit W(i) == 1
                    # (i.e. they are different)
                    # quantize to the nearest even number
                    # otherwise quantize to the nearest odd number

                    original_val = block[bit_i]
                    original_val_int = int(block[bit_i])

                    # this is needed in case the coefficient is negative since
                    # we ignore negativity during quantization
                    was_negative = False
                    if original_val_int >= 0:
                        original_val_bin = format(original_val_int,
                                'b').zfill(32)
                    else:
                        original_val_bin = format(abs(original_val_int),
                                'b').zfill(32)
                        was_negative = True

                    b_prime = original_val_bin[-3]
                    w_i = bin_message[current_index]

                    if int(b_prime) ^ int(w_i):
                        # if XOR == 1
                        # quantize to the nearest even number
                        block[bit_i] = 2 * int(numpy.round(abs(original_val)/2.0))
                    else:
                        # quantize to the nearest odd number
                        block[bit_i] = 2 * int(numpy.round((abs(original_val)-1)/2.0))+1

                    if was_negative:
                        block[bit_i] *= -1

                    if original_val != 0 and not did_print:
                        self.debug_vals = [si, sj, bi, bj, bit_i]
                        self._debug_message(si, sj, bi, bj, bit_i)
                        self._debug_message(original_val, original_val_int,
                            original_val_bin)
                        self._debug_message(b_prime, w_i, block[bit_i])
                        self._debug_message(w_i, 'embedded')
                        self._debug_message('-'*10)
                        did_print = True

                    current_index += 1

                # un-zigzag the block
                block = block[self.zigzagflatinverse].reshape(block_w, block_h)
                # inverse 2D DCT
                block = idct(idct(block, norm='ortho').T, norm='ortho').T
                # place the watermarked block back into the slice
                ps_blocks[bi, bj] = block
            # reassemble the tile
            slices[si, sj] = numpy.hstack([numpy.vstack(ps_blocks[:,i])
                for i in range(slice_w/block_w)])

        # reassemble the slices into the watermarked image plane
        bands[embed_band] = numpy.hstack([numpy.vstack(slices[:,i])
            for i in range(2)])

        debug_str = ' '.join(bin_message)
        self._debug_message(debug_str)

        # merge channels and convert back to RGB
        watermarked = Image.merge('YCbCr', [Image.fromarray(b) for b in bands])
        return watermarked



    # 504 is the length of a 32-byte/256-bit message when
    # (63, 32) RS-encoded (504 bits = 63 bytes)
    # 120 is the length of a 8-byte message when (15, 8) encoded
    @seeded
    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)

        bin_message = ''
        extract_band = ['luma', 'cb', 'cr'].index(self.chan)
        band = bands[extract_band]
        # using nomenclature from paper
        # - slices are quadrants of the image plane being watermarked
        # - process-significant (ps) blocks are subdivisions of the slices
        #   into which watermark data is embedded
        slice_w, slice_h = width/2, height/2
        block_w, block_h = max(8, slice_w/64), max(8, slice_h/64)

        # divide image plane into slices
        slices = numpy.array(
            [band[i*slice_w:(i+1)*slice_w, j*slice_h:(j+1)*slice_h]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, slice_w, slice_h)
        slice_messages = [[],[],[],[]]

        bits_per_block = 4

        did_print = False
        # extract the message in each slice
        # for each slice
        for (si, sj) in numpy.ndindex(2, 2):
            current_index = 0
            # divide current slice into ps blocks
            ps_blocks = numpy.array(
                [slices[si,sj][i*block_w:(i+1)*block_w, j*block_h:(j+1)*block_h]
                for (i, j) in numpy.ndindex(slice_w/block_w, slice_h/block_h)]
            ).reshape(slice_w/block_w, slice_h/block_h, block_w, block_h)

            # for each block
            for (bi, bj) in numpy.ndindex(slice_w/block_w, slice_h/block_h):
                if current_index >= message_length:
                    break
                # 2D DCT
                block = dct(dct(ps_blocks[bi,bj].T,norm='ortho').T,norm='ortho')
                block = block.flatten()[self.zigzagflat] # get in zig zag order

                # extract <bits_per_block> bits of the message from this block
                for bit_i in range(1, 1+bits_per_block):
                    val = block[bit_i]
                    val_int = int(val)
                    was_negative = False
                    if val_int >= 0:
                        val_bin = format(val_int, 'b').zfill(32)
                    else:
                        val_bin = format(abs(val_int), 'b').zfill(32)
                        was_negative = True
                    first_bit = val_bin[-1]
                    third_bit = val_bin[-3]
                    if int(first_bit, 2) ^ int(third_bit, 2):
                        slice_messages[si*2+sj].append(0)
                        extracted_bit = 0
                    else:
                        slice_messages[si*2+sj].append(1)
                        extracted_bit = 1

                    if self.debug and \
                       self.debug_vals == [si, sj, bi, bj, bit_i] and \
                       not did_print:
                        self._debug_message(si, sj, bi, bj, bit_i)
                        self._debug_message(val, val_int, val_bin)
                        self._debug_message(third_bit, first_bit)
                        self._debug_message(extracted_bit, 'extracted')
                        self._debug_message('-'*10)
                        did_print = True

                    current_index += 1

        # check the bits we got from the slices
        debug_str = ''
        for s in range(message_length):
            total = sum([slice_messages[i][s] for i in range(4)])
            debug_str += str(total) + ' '
            if total > 2:
                bin_message += '1'
            else:
                bin_message += '0'
        self._debug_message(debug_str)

        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])
