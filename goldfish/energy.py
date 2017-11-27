from watermarker import *
from scipy.fftpack import dct, idct
from unireedsolomon import rs

class EnergyWatermarker(Watermarker):
    '''
    Embedder/extractor using Lee/Li's pre-inserted code method from
    J. Lee and B. Li. Self-Recognized Image Protection Technique that Resists
    Large-Scale Cropping. 2014.
    '''
    def __init__(self, k=33, m=5, **kwargs):
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
        self.k = k # what are these two?
        self.m = m

    @seeded
    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size

        #convert to YCbCr to get luma
        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)

        # encode with Reed-Solomon
        # message is 32 bytes
        coder = rs.RSCoder(63, 32)
        field_vals = coder.encode(message, return_string=False)
        self._debug_message([int(fv) for fv in field_vals])
        bin_message = ''.join([bin(fv)[2:].zfill(8) for fv in field_vals])
        self._debug_message(bin_message[:32])
        self._debug_message('Encoded message length =', len(bin_message),
                'bits')

        luma = bands[0]
        # using nomenclature from paper
        # - slices are quadrants of the image plane being watermarked
        # - process-significant (ps) blocks are subdivisions of the slices
        #   into which watermark data is embedded
        slice_w, slice_h = width/2, height/2
        block_w, block_h = max(8, slice_w/64), max(8, slice_h/64)

        # divide image plane into slices
        slices = numpy.array(
            [luma[i*slice_w:(i+1)*slice_w, j*slice_h:(j+1)*slice_h]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, slice_w, slice_h)

        bits_per_block = 4

        # embed the message in each slice
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
                if current_index >= len(bin_message):
                    break
                # 2D DCT
                block = dct(dct(ps_blocks[bi,bj].T,norm='ortho').T,norm='ortho')
                block = block.flatten()[self.zigzagflat] # get in zig zag order

                # embed <bits_per_block> bits of the message into this block
                for bit_i in range(1, 1+bits_per_block):
                    third_bit = format(int(block[bit_i]), 'b').zfill(3)[-3]
                    original_val = block[bit_i]
                    if   bin_message[current_index] == '1' and third_bit == '1':
                        block[bit_i] = 2 * int(numpy.round((block[bit_i]-1)/2.0)) + 1
                    elif bin_message[current_index] == '1' and third_bit == '0':
                        block[bit_i] = 2 * int(numpy.round((block[bit_i])/2.0))
                    elif bin_message[current_index] == '0' and third_bit == '1':
                        block[bit_i] = 2 * int(numpy.round((block[bit_i])/2.0))
                    else: # bin_message[current_index] == '0' and third_bit == '0'
                        block[bit_i] = 2 * int(numpy.round((block[bit_i]-1)/2.0)) + 1

                    # set the noticing parameter
                    d = block[self.k + bit_i]
                    d_prime = int(numpy.round(d))
                    d_bits = format(d_prime, 'b').zfill(5)

                    # handle the negative sign Python puts in
                    was_neg = False
                    if d_bits[0] == '-':
                        d_bits = d_bits[1:]
                        was_neg = True

                    if block[bit_i] == original_val:
                        # embed a 0 into the mth bit of the kth+bit_i coefficient
                        d_np = d_bits[:-self.m]+'0'+d_bits[-self.m+1:]
                    else:
                        # embed a 1
                        d_np = d_bits[:-self.m]+'1'+d_bits[-self.m+1:]

                    # replace the negative if it was there
                    if was_neg:
                        d_np = int('-'+d_np, 2)
                    else:
                        d_np = int(d_np, 2)

                    # place the watermarked coefficient back into the block
                    block[self.k + bit_i] = d_np - d_prime + d # ...yeah, sure
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
        bands[0] = numpy.hstack([numpy.vstack(slices[:,i]) for i in range(2)])

        # merge channels and convert back to RGB
        watermarked = Image.merge('YCbCr', [Image.fromarray(b) for b in bands])
        return watermarked.convert('RGB')



    # 504 is the length of a 32-byte/256-bit message when
    # (63, 32) RS-encoded
    # 120 is the length of a 8-byte message when (15, 8) encoded
    @seeded
    def extract(self, image, message_length=504):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        #convert to YCbCr to get luma
        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)

        bin_message = ''
        # get a decoder ready
        decoder = rs.RSCoder(63, 32)
        luma = bands[0]
        # using nomenclature from paper
        # - slices are quadrants of the image plane being watermarked
        # - process-significant (ps) blocks are subdivisions of the slices
        #   into which watermark data is embedded
        slice_w, slice_h = width/2, height/2
        block_w, block_h = max(8, slice_w/64), max(8, slice_h/64)

        # divide image plane into slices
        slices = numpy.array(
            [luma[i*slice_w:(i+1)*slice_w, j*slice_h:(j+1)*slice_h]
            for (i, j) in numpy.ndindex(2, 2)]
        ).reshape(2, 2, slice_w, slice_h)
        slice_messages = [[],[],[],[]]

        bits_per_block = 4

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
                    val = format(int(block[bit_i]), 'b').zfill(4)
                    first_bit = val[-1]
                    third_bit = val[-3]
                    if int(first_bit, 2) ^ int(third_bit, 2):
                        slice_messages[si*2+sj].append(0)
                    else:
                        slice_messages[si*2+sj].append(1)
                    current_index += 1

        # check the bits we got from the slices
        summed = numpy.sum(slice_messages, axis=0)
        for s in summed:
            if s > 2:
                bin_message += '1'
            else:
                bin_message += '0'

        # form the binary message
        encoded = ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])
        message = ''
        try:
            message = decoder.decode(encoded)[0]
        except rs.RSCodecError:
            print 'RSCoder failed to read encoded message!'
        return message[:32]

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            #output.append(numpy.array(list(band.getdata())))
            output[-1].resize(image.width, image.height)
        return output
