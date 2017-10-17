from watermarker import *
from scipy.fftpack import dct, idct

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
    def __init__(self, **kwargs):
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
        self.entropy_threshold = 4000
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

    @seeded
    def embed(self, image, message, quality=75, chan='red'):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        image = image.convert('YCbCr')
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        if chan == 'red':
            embed_band = 0
        if chan == 'green':
            embed_band = 1
        if chan == 'blue':
            embed_band = 2

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        quantize_matrix = self._setup_quantize_matrix(quality)
        
        bw = 8 # block width
        bh = 8 # block height

        # embed in the blue band
        band = bands[embed_band]

        entropies = []
        bits_per_block = 16
        current_index = 0
        # divide the tile into 8x8 ps blocks
        blocks = numpy.array(
            [band[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
            for (i, j) in numpy.ndindex(width/bw, width/bh)]
        ).reshape(width/bw, height/bh, bw, bh)
        # for each block
        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            if current_index >= len(bin_message):
                #blocks[bi, bj] = 255
                break
            # 2d dct
            block = dct(dct(blocks[bi, bj].T, norm='ortho').T,
                    norm='ortho')
            # check block entropy
            orig_entropy = numpy.sum(block*block) - block[0,0]*block[0,0]
            entropies.append(orig_entropy)
            if orig_entropy < self.entropy_threshold:
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
            # check the new entropy
            new_entropy = numpy.sum(block*block) - block[0,0]*block[0,0]
            if new_entropy < self.entropy_threshold:
                #blocks[bi, bj][:] = 64
                continue # leave this block as is and continue
            else:
                current_index += bits_per_block
            # multiply by quantization matrix
            block *= quantize_matrix
            # inverse dct
            block = idct(idct(block, norm='ortho').T, norm='ortho').T
            # reassign back to tile
            blocks[bi, bj] = block
        if current_index < len(bin_message):
            print 'Did not finish'
            print current_index, 'out of', len(bin_message), 'embedded'
        # reassemble the tile
        band = numpy.hstack([numpy.vstack(blocks[:,i])
            for i in range(width/bw)])

        print min(entropies), max(entropies), sum(entropies)/len(entropies)
        print sorted(entropies, reverse=True)[:5]

        # reassemble the tiles into a channel
        bands[embed_band] = band
        #numpy.hstack([numpy.vstack(tiles[:, i]) for i in range(2)])

        #return Image.merge('RGB', [Image.fromarray(b) for b in bands])
        return Image.merge('YCbCr', [Image.fromarray(b) for b in bands])
        #return Image.fromarray(band)

    @seeded
    def extract(self, image, message_length=256, quality=75, chan='red'):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        image = image.convert('YCbCr')
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        if chan == 'red':
            ex_band = 0
        if chan == 'green':
            ex_band = 1
        if chan == 'blue':
            ex_band = 2

        bin_message = ''
        quantize_matrix = self._setup_quantize_matrix(quality)
        
        bw = 8 # block width
        bh = 8 # block height

        band = bands[ex_band]

        # embed the message in each tile
        bits_per_block = 16
        current_index = 0
        # divide the tile into 8x8 ps blocks
        blocks = numpy.array(
            [band[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
            for (i, j) in numpy.ndindex(width/bw, height/bh)]
        ).reshape(width/bw, height/bh, bw, bh)
        # for each block
        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            if current_index >= message_length:
                break
            # 2d dct
            block = dct(dct(blocks[bi, bj].T, norm='ortho').T,
                    norm='ortho')
            block /= quantize_matrix
            entropy = numpy.sum(block*block) - block[0,0]*block[0,0]
            if entropy < self.entropy_threshold:
                continue
            block = block.flatten()[self.zigzagflat] # get in zig zag order
            for bit_i in range(1, 1+bits_per_block):
                if numpy.round(block[bit_i]) % 2 == 0:
                    bin_message += '0'
                else:
                    bin_message += '1'
            current_index += bits_per_block

        bin_message = bin_message[:256]
        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _setup_quantize_matrix(self, quality):
        if quality < 50:
            s = 5000/quality
        else:
            s = 200 - 2*quality

        return numpy.floor((s*self.base_quantize_matrix + 50) / 100)
