from watermarker import *
from scipy.fftpack import dct, idct
from unireedsolomon import rs

# TODO:
# These functions originally were an implementation of Bo Li's method
# and still contain remnants of that code.
# embed() and extract() need to be cleaned up to remove these, since
# that method is going to be used in another module
class EntropyWatermarker(Watermarker):
    '''
    Embedder/extractor using the entropy thresholding scheme from
    K. Solanki. Multimedia Data Hiding, 2006.
    Section 3.3.1
    '''
    def __init__(self):
        Watermarker.__init__(self)
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

    @seeded
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
        tw = width/1  # tile width
        th = height/1 # tile height
        psw = 8       # process-significant block width
        psh = 8       # process-significant block height
        '''
        tiles = numpy.array(
            [bands[2][i*tw:(i+1)*tw, j*th:(j+1)*th]
            for (i, j) in numpy.ndindex(1, 1)]
        ).reshape(1, 1, tw, th)
        '''
        band = bands[2]

        # embed the message in each tile
        bits_per_block = 4
        # for each tile
        for (ti, tj) in numpy.ndindex(1, 1):
            current_index = 0
            # divide the tile into 8x8 ps blocks
            ps_blocks = numpy.array(
                [band[i*psw:(i+1)*psw, j*psh:(j+1)*psh]
                for (i, j) in numpy.ndindex(width/psw, width/psh)]
            ).reshape(tw/psw, th/psh, psw, psh)
            # for each block
            for (bi, bj) in numpy.ndindex(tw/psw, th/psh):
                if current_index >= len(bin_message):
                    ps_blocks[bi, bj] = 255
                    break
                # 2d dct
                block = dct(dct(ps_blocks[bi, bj].T, norm='ortho').T,
                        norm='ortho')
                # check block energy
                orig_energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                if orig_energy < self.energy_threshold:
                    ps_blocks[bi, bj][:] = 128
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
                # un-zigzag the block
                block = block[self.zigzagflatinverse].reshape(8, 8)
                # check the new energy
                new_energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                if new_energy < self.energy_threshold:
                    ps_blocks[bi, bj][:] = 64
                    # continue # skip this block
                else:
                    current_index += bits_per_block
                # multiply by quantization matrix
                block *= quantize_matrix
                # inverse dct
                block = idct(idct(block, norm='ortho').T, norm='ortho').T
                # reassign back to tile
                ps_blocks[bi, bj] = block
            # reassemble the tile
            band = numpy.hstack([numpy.vstack(ps_blocks[:,i])
                for i in range(tw/psw)])

        # reassemble the tiles into a channel
        bands[2] = band
        #numpy.hstack([numpy.vstack(tiles[:, i]) for i in range(2)])

        return Image.merge('RGB', [Image.fromarray(b) for b in bands])

    @seeded
    def extract(self, image, message_length=256, quality=75):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        bin_message = ''
        quantize_matrix = self._setup_quantize_matrix(quality)
        
        # just deal with the red band for now
        # split into quarter tiles
        # terrible variable names ahead, deal with it
        tw = width/1  # tile width
        th = height/1 # tile height
        psw = 8       # process-significant block width
        psh = 8       # process-significant block height
        '''
        tiles = numpy.array(
            [bands[2][i*tw:(i+1)*tw, j*th:(j+1)*th]
            for (i, j) in numpy.ndindex(1, 1)]
        ).reshape(1, 1, tw, th)
        '''
        band = bands[0]
        tile_messages = [[],[],[],[]]

        # embed the message in each tile
        bits_per_block = 4
        # for each tile
        for (ti, tj) in numpy.ndindex(2, 2):
            #if not (ti == 1 and tj == 0):
            #    continue
            current_index = 0
            # divide the tile into 8x8 ps blocks
            ps_blocks = numpy.array(
                [band[i*psw:(i+1)*psw, j*psh:(j+1)*psh]
                for (i, j) in numpy.ndindex(tw/psw, th/psh)]
            ).reshape(tw/psw, th/psh, psw, psh)
            # for each block
            for (bi, bj) in numpy.ndindex(tw/psw, th/psh):
                if current_index >= message_length:
                    break
                # 2d dct
                block = dct(dct(ps_blocks[bi, bj].T, norm='ortho').T,
                        norm='ortho')
                block /= quantize_matrix
                energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                if energy < self.energy_threshold:
                    continue
                block = block.flatten()[self.zigzagflat] # get in zig zag order
                for bit_i in range(1, 1+bits_per_block):
                    if numpy.round(block[bit_i]) % 2 == 0:
                        bin_message += '0'
                    else:
                        bin_message += '1'

        bin_message = bin_message[:256]
        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _setup_quantize_matrix(self, quality):
        if quality < 50:
            s = 5000/quality
        else:
            s = 200 - 2*quality

        return numpy.floor((s*self.base_quantize_matrix + 50) / 100)
