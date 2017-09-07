import sys
sys.path.append('../')
from goldfish.histogram import HistogramWatermarker
import uuid

from PIL import Image, ImageMath

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
outfile = infile[:-4]+'-altered.png'

message = uuid.uuid4().hex
wm = HistogramWatermarker()

print 'Embedding message \"'+message+'\" into image'

im_out = wm.embed(infile, message)

print 'Saving to', outfile

im_out.save(outfile)

print 'Loading from', outfile, 'to retrieve message'

retrieved = wm.extract(im_out) #outfile)

if message != retrieved:
    print 'Failure!'
    print message
    print retrieved
else:
    print 'Success!'

wm.show_plot()

print 'Getting the diff'
im_in = Image.open(infile)
in_bands = im_in.split()
out_bands = im_out.split()
diffs = [ImageMath.eval("convert(b-a, 'L')", a=in_bands[i], b=out_bands[i])
        for i in range(len(in_bands))]
diff = Image.merge('RGB', diffs)
diff.save('diff.png')
