'''
from goldfish.Embedders import LSBEmbedder
from goldfish.Extractors import LSBExtractor
'''
import sys
sys.path.append('../')
from goldfish.watermarker import Watermarker
import uuid

from PIL import Image, ImageMath

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
outfile = infile[:-4]+'-altered.jpg'

message = uuid.uuid4().hex
wm = Watermarker()

print 'Embedding message \"'+message+'\" into image'

im_out = wm.embed(infile, message)

# only reading back from image in memory
# LSB doesn't survive most image compression
retrieved = wm.extract(im_out)

if message != retrieved:
    print 'Failure!'
    print message
    print retrieved
else:
    print 'Success!'

print 'Getting the diff'
im_in = Image.open(infile)
in_bands = im_in.split()
out_bands = im_out.split()
diffs = [ImageMath.eval("convert(a - b, 'L')", a=in_bands[i], b=out_bands[i])
        for i in range(len(in_bands))]
diff = Image.merge('RGB', diffs)
diff.save('diff.png')
