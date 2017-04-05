import sys
sys.path.append('../')
from goldfish.Embedders import HistogramEmbedder
from goldfish.Extractors import HistogramExtractor
import uuid

from PIL import Image, ImageMath

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
outfile = ''.join(infile.split('.')[:-1])+'-altered.png'

message = uuid.uuid4().hex
em = HistogramEmbedder()

print 'Embedding message \"'+message+'\" into image'

im_out = em.embed(infile, message)

print 'Saving to', outfile

im_out.save(outfile)

print 'Loading from', outfile, 'to retrieve message'

ex = HistogramExtractor()

retrieved = ex.extract(outfile)

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
diffs = [ImageMath.eval("convert(b-a, 'L')", a=in_bands[i], b=out_bands[i])
        for i in range(len(in_bands))]
diff = Image.merge('RGB', diffs)
diff.save('diff.png')
