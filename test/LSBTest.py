import sys
sys.path.append('../')
from goldfish.Embedders import LSBEmbedder
from goldfish.Extractors import LSBExtractor
import uuid

from PIL import Image

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
outfile = ''.join(infile.split('.')[:-1])+'-altered.png'

message = uuid.uuid4().hex
im = Image.open(infile)
em = LSBEmbedder()

print 'Embedding message \"'+message+'\" into image'

im_out = em.embed(im, message)

print 'Saving to', outfile

im_out.save(outfile)

print 'Loading from', outfile, 'to retrieve message'

im = Image.open(outfile)
ex = LSBExtractor()

retrieved = ex.extract(im)

if message != retrieved:
    print 'Failure!'
    print message
    print retrieved
else:
    print 'Success!'

