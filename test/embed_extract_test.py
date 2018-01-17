import sys
sys.path.append('../')
from goldfish.watermarker import Watermarker
from goldfish.histogram import HistogramWatermarker
from goldfish.entropy import EntropyWatermarker
from goldfish.energy import EnergyWatermarker

import uuid
import os

from collections import Counter
#from PIL import Image, ImageMath
#from matplotlib import pyplot

if len(sys.argv) == 1:
    print 'Usage:', sys.argv[0], '<directory>',
    print '[<directory2> <directory3> ... ]'
    sys.exit(1)

directories = sys.argv[1:]

watermarkers = {
        'lsb' : Watermarker(),
        'hist' : HistogramWatermarker(),
        'entropy' : EntropyWatermarker(),
        'energy' : EnergyWatermarker()
        }
scores = Counter()

n_total = 0
n_done = 0

for directory in directories:
    files = os.listdir(directory)
    files = [f for f in files if not f.startswith('.') and 'altered' not in f]
    n_total += len(files)
n_total *= len(watermarkers)
len_total = len(str(n_total))

sys.stdout.write('0')
sys.stdout.flush()
for directory in directories:
    files = os.listdir(directory)
    for filename in files:
        if filename.startswith('.'):
            continue

        for watermark_type in watermarkers:
            sys.stdout.write('\r'+str(n_done).rjust(len_total)+'/'+str(n_total))
            sys.stdout.flush()
            marker = watermarkers[watermark_type]
            save_before_extract = True
            if watermark_type == 'lsb':
                save_before_extract = False
            
            message = uuid.uuid4().hex
            im_out = marker.embed(directory+filename, message)
            retrieved = ''

            if save_before_extract:
                output_name = directory+filename[:-4] + 'altered.jpg'
                im_out.save(output_name, quality=95)
                retrieved = marker.extract(output_name)
                os.remove(output_name)
            else:
                retrieved = marker.extract(im_out)

            if retrieved == message:
                scores[watermark_type] += 1

            n_done += 1
sys.stdout.write('\r'+str(n_done).rjust(len_total)+'/'+str(n_total)+'\n')
sys.stdout.flush()

for watermarker_type in scores:
    print watermarker_type, scores[watermarker_type]
