from watermarker import *
from matplotlib import pyplot

class HistogramWatermarker(Watermarker):
    '''
    Embedder/extractor using the reversible histogram method from
    Z. Ni, Y. Shi, et al. Reversible data hiding, 2006
    '''
    def __init__(self, chan='R', **kwargs):
        Watermarker.__init__(self, **kwargs)
        if self.debug:
            self.figure, self.axes = pyplot.subplots(2,sharex=True,tight_layout=True)
        else:
            self.figure = None
        self.chan = chan

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

        embed_band = 'RGB'.index(self.chan)
        # get the histogram and alter the image channels
        hist = Image.fromarray(bands[embed_band]).histogram()
        if self.debug:
            self.axes[0].set_title('before')
            self.axes[0].bar(range(256), hist, color=self.chan.lower(), alpha=0.5)
        peak = self._get_max_point(hist)
        trough = self._get_min_point(hist)
        self._debug_message(peak)
        self._debug_message(trough)
        bin_message = ''.join([format(ord(c), 'b').zfill(8) for c in message])
        self._debug_message(bin_message[:32])
        # we can shift values to the right and embed the data at the same time
        current_index = 0
        done_embedding = False
        # for each band, shift right and embed data if we are not done and if
        # this is the right band for the current bit
        for i in range(width):
            for j in range(height):
                if peak <= bands[embed_band][i,j] <= trough:
                    bands[embed_band][i,j] += 1
                elif bands[embed_band][i,j] == peak-1 and not done_embedding:
                    if current_index >= len(bin_message):
                        done_embedding = True
                    else:
                        bands[embed_band][i,j] += int(bin_message[current_index])
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

        embed_band = 'RGB'.index(self.chan)
        hist = Image.fromarray(bands[embed_band]).histogram()
        if self.debug:
            self.axes[1].set_title('after')
            self.axes[1].bar(range(256), hist, color=self.chan.lower(), alpha=0.5)
            pyplot.show()
        peak = self._get_max_point(hist)
        trough = self._get_min_point(hist)
        self._debug_message(peak)
        self._debug_message(trough)
        bin_message = ''

        current_index = 0
        done_extracting = False
        for i in range(width):
            if done_extracting:
                break
            for j in range(height):
                if done_extracting:
                    break
                if bands[embed_band][i,j] == peak - 2:
                    bin_message += '0'
                    current_index += 1
                elif bands[embed_band][i,j] == peak - 1:
                    bin_message += '1'
                    current_index += 1
                if current_index >= message_length:
                    done_extracting = True
        self._debug_message(bin_message[:32])
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
