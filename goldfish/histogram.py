from watermarker import *
from matplotlib import pyplot

class HistogramWatermarker(Watermarker):
    '''
    Embedder/extractor using the reversible histogram method from
    Z. Ni, Y. Shi, et al. Reversible data hiding, 2006
    '''
    def __init__(self):
        Watermarker.__init__(self)
        self.figure, self.axes = pyplot.subplots(2,sharex=True,tight_layout=True)

    def show_plot(self):
        if self.figure is None:
            return
        pyplot.show()

    @seeded
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
        self.axes[0].set_title('before')
        self.axes[0].bar(range(256), hists[0], color='r', alpha=0.5)
        self.axes[0].bar(range(256), hists[1], color='g', alpha=0.5)
        self.axes[0].bar(range(256), hists[2], color='b', alpha=0.5)
        #pyplot.show()
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

    @seeded
    def extract(self, image, message_length=256):
        if type(image) is str or type(image) is unicode:
            image = Image.open(image)

        width, height = image.size
        if len(image.getbands()) > 3:
            bands = self._get_bands(image[:3])
        else:
            bands = self._get_bands(image)

        hists = [Image.fromarray(b).histogram() for b in bands]
        self.axes[1].set_title('after')
        self.axes[1].bar(range(256), hists[0], color='r', alpha=0.5)
        self.axes[1].bar(range(256), hists[1], color='g', alpha=0.5)
        self.axes[1].bar(range(256), hists[2], color='b', alpha=0.5)
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
                    #print 'bands['+str(chan)+']['+str(i)+','+str(j)+'] =',
                    #print bands[chan][i, j],
                    #print 'peaks['+str(chan)+'] =', peaks[chan]
                    bin_message += '1'
                    current_index += 1
                if current_index >= message_length:
                    done_extracting = True
        print bin_message[:32]
        return ''.join([chr(int(bin_message[i:i+8], 2))
            for i in range(0, len(bin_message), 8)])

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
