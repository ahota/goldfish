from watermarker import *
from scipy.fftpack import dct, idct
from unireedsolomon import rs

class EnergyWatermarker(Watermarker):
    '''
    Embedder/extractor using the entropy thresholding scheme from
    K. Solanki. Multimedia Data Hiding, 2006.
    Section 3.3.1
    '''
    def __init__(self, energy=1000, block_size=(8, 8)):
        self.energy_threshold = energy
        self.block_size = block_size
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

    def embed(self, image, message):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        rgb = self._get_bands(image)
        '''
        for (i, j) in numpy.ndindex(width, height):
            if (rgb[:, i, j] < 5).all():
                rgb[:, i, j] = 5
        '''
        image = Image.merge('RGB', [Image.fromarray(rgb[b]) for b in range(3)])

        #convert to YCbCr to get luma
        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)

        # encode with Reed-Solomon
        # message is 32 bytes
        coder = rs.RSCoder(63, 32)
        field_vals = coder.encode(message, return_string=False)
        #print [int(fv) for fv in field_vals]
        bin_message = ''.join([bin(fv)[2:].zfill(8) for fv in field_vals])
        #print bin_message[:32]
        #print 'Encoded message length =', len(bin_message), 'bits'

        luma = bands[0]
        bw, bh = self.block_size
        blocks = numpy.array([
            luma[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
            for (i, j) in numpy.ndindex(width/bw, height/bh)
        ]).reshape(width/bw, height/bh, bw, bh)
        quantize_matrix = self._setup_quantize_matrix(75)
        num_usable = 0
        current_index = 0
        bits_per_block = 4
        delta = 3.0

        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            block = blocks[bi, bj]
            block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            energy = numpy.sum(block*block) - block[0,0]*block[0,0]
            if energy > self.energy_threshold:
                block /= quantize_matrix
                block = block.flatten()[self.zigzagflat]
                if current_index < len(bin_message):
                    for bit_i in range(1, 1+bits_per_block):
                        if current_index + bit_i - 1 >= len(bin_message):
                            current_index = len(bin_message)
                            rgb[0][bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh] = \
                                numpy.minimum(
                                rgb[0][bi*bw:(bi+1)*bw,
                                    bj*bh:(bj+1)*bh].astype(int)+64,
                                numpy.ones((bw, bh))*255)
                            continue
                        if bin_message[current_index + bit_i - 1] == '1':
                            block[bit_i] = delta * numpy.round(
                                    (block[bit_i]+delta/2)/delta) - delta/2
                        else:
                            block[bit_i] = delta * numpy.round(
                                    block[bit_i]/delta)
                    # undo the zigzag and JPEG quantization to check the new
                    # energy level
                    block = block[self.zigzagflatinverse]
                    block = block.reshape(self.block_size) * quantize_matrix
                    new_energy = numpy.sum(block*block) - block[0,0]*block[0,0]
                    if new_energy < self.energy_threshold:
                        # this block can't be used
                        # but we need to keep the data in it to stay low
                        # darken it in the rgb to show
                        for band in rgb:
                            band[bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh] = \
                                numpy.maximum(
                                band[bi*bw:(bi+1)*bw, 
                                    bj*bh:(bj+1)*bh].astype(int)-64,
                                numpy.zeros((bw, bh)))
                    else:
                        # this block can be used
                        # lighten it in the rgb to show
                        num_usable += 1
                        current_index += bits_per_block
                        for band in rgb:
                            band[bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh] = \
                                numpy.minimum(
                                band[bi*bw:(bi+1)*bw,
                                    bj*bh:(bj+1)*bh].astype(int)+32,
                                numpy.ones((bw, bh))*255)
                    # this has to be done, even for blocks whose energy dropped
                    # too low, so that the decoder will only look at blocks with
                    # high energy + data
                    block = idct(idct(block,norm='ortho').T,norm='ortho').T
                    blocks[bi, bj] = block
                else:
                    break
                    # these blocks were scanned after the message has been
                    # embedded, brighten the red channel
                    rgb[0][bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh] = \
                        numpy.minimum(
                        rgb[0][bi*bw:(bi+1)*bw,bj*bh:(bj+1)*bh].astype(int)+64,
                        numpy.ones((bw, bh))*255)

        # join the blocks
        luma = numpy.hstack([numpy.vstack(blocks[:, j])
            for j in range(height/bh)])
        bands[0] = luma
        #print num_usable, 'blocks'
        #print current_index, 'of', len(bin_message), 'bits embedded'
        # show the modified rgb to indicate block status
        #Image.merge('RGB', [Image.fromarray(band) for band in rgb]).show()
        return Image.merge('YCbCr',
                [Image.fromarray(band) for band in bands]).convert('RGB')

    # 504 is the length of a 32-byte/256-bit message when
    # (63, 32) RS-encoded
    # 120 is the length of a 8-byte message when (15, 8) encoded
    def extract(self, image, message_length=504):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        rgb = self._get_bands(image)
        #convert to YCbCr to get luma
        yimage = image.convert('YCbCr')
        bands = self._get_bands(yimage)
        #bands = self._get_bands(image)

        bin_message = ''
        # get a decoder ready
        decoder = rs.RSCoder(63, 32)
        luma = bands[0]
        bw, bh = self.block_size
        blocks = numpy.array([
            luma[i*bw:(i+1)*bw, j*bh:(j+1)*bh]
            for (i, j) in numpy.ndindex(width/bw, height/bh)
        ]).reshape(width/bw, height/bh, bw, bh)
        quantization_matrix = self._setup_quantize_matrix(75)
        current_index = 0
        bits_per_block = 4
        delta = 3.0

        for (bi, bj) in numpy.ndindex(width/bw, height/bh):
            block = blocks[bi, bj]
            block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            energy = numpy.sum(block*block) - block[0,0]*block[0,0]
            if energy > self.energy_threshold:
                block /= quantization_matrix
                block = block.flatten()[self.zigzagflat]
                if current_index < message_length:
                    for band in rgb:
                        band[bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh] = \
                            numpy.minimum(
                            band[bi*bw:(bi+1)*bw, bj*bh:(bj+1)*bh]+32,
                            numpy.ones((bw, bh))*255)
                    coeffs = []
                    for bit_i in range(1, 1+bits_per_block):
                        if numpy.round(block[bit_i]) % delta == 0:
                            bin_message += '0'
                        else:
                            bin_message += '1'
                        coeffs.append(numpy.round(block[bit_i]))
                    #print coeffs
                    current_index += bits_per_block
                else:
                    break

        #Image.fromarray(rgb[0]).show()
        #Image.merge('RGB', [Image.fromarray(band) for band in rgb]).show()
        #print bin_message[:32]
        encoded = ''.join([chr(int(bin_message[i:i+8], 2)) 
                 for i in range(0, len(bin_message), 8)])
        try:
            message = decoder.decode(encoded)[0]
        except rs.RSCodecError:
            message = 'oh noes watermark didnt work! :('
        return message[:32] #[:8]

    def _get_bands(self, image):
        bands = image.split()
        output = []
        for band in bands:
            output.append(numpy.fromiter(iter(band.getdata()), numpy.uint8))
            output[-1].resize(image.width, image.height)
        return output

    def _setup_quantize_matrix(self, quality):
        base_quantize_matrix = numpy.array(
                [[16,  11,  10,  16,  24,  40,  51,  61],
                 [12,  12,  14,  19,  26,  58,  60,  55],
                 [14,  13,  16,  24,  40,  57,  69,  56],
                 [14,  17,  22,  29,  51,  87,  80,  62],
                 [18,  22,  37,  56,  68, 109, 103,  77],
                 [24,  35,  55,  64,  81, 104, 113,  92],
                 [49,  64,  78,  87, 103, 121, 120, 101],
                 [72,  92,  95,  98, 112, 100, 103,  99]]
        )
        if quality < 50:
            s = 5000/quality
        else:
            s = 200 - 2*quality

        return numpy.floor((s*base_quantize_matrix + 50) / 100)
