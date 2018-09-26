from watermarker import *
from scipy.fftpack import dct, idct
from unireedsolomon import rs

from matplotlib import pyplot

class EntropyWatermarker(Watermarker):
    '''
    Embedder/extractor using the entropy thresholding scheme from
    K. Solanki. Multimedia Data Hiding, 2006.
    Section 3.3.1
    '''
    def __init__(self, quality=75, bits=16, threshold=4000, chan='luma',
            show_embed=False, **kwargs):
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
        self.quality = quality
        self.bits_per_block = bits
        self.entropy_threshold = threshold
        self.chan = chan
        self.show_embed = show_embed
        # heatmaps of pre- and post- embedding entropy
        self.fig, self.axes = pyplot.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        self.fig.set_size_inches((13.3, 5))
        self.max_entropy = 0

    def _divide_blocks(self, band, width, height, bw, bh):
        return numpy.array(
                [band[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
                for (i, j) in numpy.ndindex(width/bw, width/bh)]
                ).reshape(width/bw, height/bh, bw, bh)

    def _calculate_entropy(self, block):
        return numpy.sum(block * block) - block[0, 0]*block[0, 0]

    def _embed_k_bits(self, block, bin_message, current_index):
        # embed bits_per_block bits of the message into this block
        for bit_i in range(1, 1+self.bits_per_block):
            if current_index + bit_i - 1 >= len(bin_message):
                self._debug_message('Early termination:',
                    'embedded length of message at block',
                    bi, bj, 'and bit', bit_i)
                break
            if bin_message[current_index+bit_i-1] == '1':
                # round coefficient to the nearest odd number
                block[bit_i] = 2 * numpy.round((block[bit_i]+1)/2) - 1
            else:
                # round coefficient to the nearest even number
                block[bit_i] = 2 * numpy.round(block[bit_i]/2)

    def _prep_image(self, image):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)
        width, height = image.size
        image = image.convert('YCbCr')
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)
        return (bands, width, height)

    def entropy_heatmap(self, image, message):
        bands, width, height = self._prepare_image(image)

        bw = 8
        bh = 8

        band = bands[0]
        blocks = self._divide_blocks(band, width, height, bw, bh)
        entropy_matrix = numpy.zeros((width/bw, height/bh))

        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            block = dct(dct(blocks[bi, bj].T, norm='ortho').T, norm='ortho')
            entropy_matrix[bi, bj] = self._calculate_entropy(block)
            block = idct(idct(block, norm='ortho').T, norm='ortho').T
            blocks[bi, bj] = block

        band = numpy.hstack([numpy.vstack(blocks[:,i]) for i in range(width/bw)])
        bands[0] = band
        merged = Image.merge('YCbCr', [Image.fromarray(b) for b in bands])
        return (merged.convert('RGB'), entropy_matrix)

    @seeded
    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        image = image.convert('YCbCr')
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)
        embed_band = ['luma', 'cb', 'cr'].index(self.chan)

        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        self._debug_message('to embed', ' '.join(bin_message))
        quantize_matrix = self._setup_quantize_matrix(self.quality)
        
        bw = 8 # block width
        bh = 8 # block height

        band = bands[embed_band]

        entropies = []
        current_index = 0
        n_blocks_embedded = 0

        # divide the tile into 8x8 blocks
        blocks = self._divide_blocks(band, width, height, bw, bh)

        # for making a heatmap
        entropy_matrix = numpy.zeros((width/bw, height/bh))
        valid_blocks = 0

        # for each block
        for (bi, bj) in numpy.ndindex(width/bw, height/bh):

            # perform 2d dct
            block = dct(dct(blocks[bi, bj].T, norm='ortho').T, norm='ortho')

            # check block entropy
            orig_entropy = self._calculate_entropy(block)
            entropies.append(orig_entropy)
            entropy_matrix[bi, bj] = orig_entropy

            if orig_entropy < self.entropy_threshold:
                continue # skip this block
            else:
                valid_blocks += 1

            if current_index >= len(bin_message):
                # we've already embedded the whole message
                continue

            # divide by the jpeg quantization matrix
            # image should resist up to <quality> jpeg compression
            block /= quantize_matrix
            block = block.flatten()[self.zigzagflat] # get in zig zag order

            self._embed_k_bits(block, bin_message, current_index)

            # un-zigzag the block
            block = block[self.zigzagflatinverse].reshape(8, 8)
            # multiply by quantization matrix
            block *= quantize_matrix

            # check the new entropy
            new_entropy = numpy.sum(block*block) - block[0,0]*block[0,0]
            if new_entropy < self.entropy_threshold:
                if self.show_embed:
                    blocks[bi, bj] = self._draw_line(blocks[bi, bj])
                # leave this block as is and continue
                self._debug_message('block {:2d} {:2d}'.format(bi, bj),
                        'fell to {:.3f} from {:.3f}'.format(new_entropy, orig_entropy))
                continue
            else:
                current_index += self.bits_per_block
                n_blocks_embedded += 1

            # inverse dct
            block = idct(idct(block, norm='ortho').T, norm='ortho').T
            # reassign back to tile
            blocks[bi, bj] = block
        # end for

        if current_index < len(bin_message):
            self._debug_message('Did not finish')
            self._debug_message(current_index, 'out of', len(bin_message), 'embedded')
        # reassemble the tile
        band = numpy.hstack([numpy.vstack(blocks[:,i])
            for i in range(width/bw)])

        # create heatmap
        im = self.axes[0].imshow(entropy_matrix, cmap='viridis', aspect=1)
        self.axes[0].set(adjustable='box-forced', aspect='equal')
        cbar = self.axes[0].figure.colorbar(im, ax=self.axes[0])
        cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

        self._debug_message('number of blocks above', self.entropy_threshold,
                valid_blocks)
        self._debug_message('min/max/avg entropy', min(entropies), max(entropies),
                sum(entropies)/len(entropies))
        self._debug_message('blocks used:', n_blocks_embedded)
        self.max_entropy = max(entropies)

        # reassemble the tiles into a channel
        bands[embed_band] = band
        watermarked = Image.merge('YCbCr', [Image.fromarray(b) for b in bands])
        return watermarked.convert('RGB')

    # message length in units of bits, not bytes!
    @seeded
    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        image = image.convert('YCbCr')
        if len(image.getbands()) > 3:
            bands = self._get_bands(image)[:3]
        else:
            bands = self._get_bands(image)

        ex_band = ['luma', 'cb', 'cr'].index(self.chan)

        bin_message = ''
        quantize_matrix = self._setup_quantize_matrix(self.quality)
        
        bw = 8 # block width
        bh = 8 # block height

        band = bands[ex_band]
        entropy_matrix = numpy.zeros((width/bw, height/bh))

        # embed the message in each tile
        current_index = 0
        # divide the tile into 8x8 ps blocks
        blocks = numpy.array(
            [band[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
            for (i, j) in numpy.ndindex(width/bw, height/bh)]
        ).reshape(width/bw, height/bh, bw, bh)
        # for each block
        valid_blocks = 0
        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            if current_index >= message_length:
                break
            # 2d dct
            block = dct(dct(blocks[bi, bj].T, norm='ortho').T,
                    norm='ortho')
            block /= quantize_matrix
            entropy = self._calculate_entropy(block)
            entropy_matrix[bi, bj] = entropy
            if entropy < self.entropy_threshold:
                continue
            valid_blocks += 1
            block = block.flatten()[self.zigzagflat] # get in zig zag order
            for bit_i in range(1, 1+self.bits_per_block):
                if current_index + bit_i - 1 >= message_length:
                    self._debug_message('Early termination:',
                        'extracted length of message at block',
                        bi, bj, 'and bit', bit_i)
                    break
                if numpy.round(block[bit_i]) % 2 == 0:
                    bin_message += '0'
                else:
                    bin_message += '1'
            current_index += self.bits_per_block


        im = self.axes[1].imshow(entropy_matrix, cmap='viridis', vmax=self.max_entropy, aspect=1)
        self.axes[1].set(adjustable='box-forced', aspect='equal')
        cbar = self.axes[1].figure.colorbar(im, ax=self.axes[1])
        cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

        self._debug_message('extracted', ' '.join(bin_message))
        self._debug_message('num bits extracted', len(bin_message))
        self._debug_message('num blocks looked at', valid_blocks)
        self._debug_message('min/max/avg entropy', numpy.min(entropy_matrix),
                numpy.max(entropy_matrix), numpy.average(entropy_matrix))
        pyplot.show()
        return ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])

    def _setup_quantize_matrix(self, quality):
        if quality < 50:
            s = 5000/quality
        else:
            s = 200 - 2*quality

        return numpy.floor((s*self.base_quantize_matrix + 50) / 100)

    def _draw_x(self, block):
        w, h = block.shape
        block[:] = 32
        for i in range(w):
            block[i, i] = 0
            block[w-1-i, i] = 0
        return block

    def _draw_line(self, block):
        w, h = block.shape
        block[:] = 64
        for i in range(w):
            block[w/2, i] = 0
        return block
