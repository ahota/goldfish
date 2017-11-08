import sys
sys.path.append('../')
from goldfish.entropy import EntropyWatermarker
import uuid

from PIL import Image, ImageMath
from skimage import data

from argparse import ArgumentParser

imagetypes = ('jpg', 'png')

parser = ArgumentParser()
parser.add_argument('image', help='Path to the image to test with')
parser.add_argument('-n', '--n-rounds', type=int,
        help='Number of rounds to perform')
parser.add_argument('-t', '--type', choices=imagetypes, default='jpg',
        help='Filetype to save during test')
parser.add_argument('-q', '--quality', type=int, default=75,
        help='Compression quality watermark should resist')
parser.add_argument('-e', '--entropy-threshold', type=int, default=4000,
        help='Block entropy threshold')
parser.add_argument('-b', '--bits-per-block', type=int, default=16,
        help='Bits per block to embed')
parser.add_argument('-c', '--channel', default='luma',
        help='Channel to embed in (luma is best)')
parser.add_argument('--debug', action='store_true',
        help='Enable debug messages')
args = parser.parse_args()

n_rounds = args.n_rounds
successes = 0

print_len = len(str(n_rounds))

for i in range(n_rounds):
    print '{num:{w}}/{t}'.format(num=i+1, w=print_len, t=n_rounds)
    sys.stdout.flush()
    infile = sys.argv[1]
    outfile = '.'.join(infile.split('.')[:-1])+'-altered.' + args.type

    message = uuid.uuid4().hex
    print len(message)
    print message
    sys.exit()
    wm = EntropyWatermarker(quality=args.quality,
            bits=args.bits_per_block,
            threshold=args.entropy_threshold,
            chan=args.channel,
            debug=args.debug)

    #print 'Embedding message \"'+message+'\" into image'

    im_out = wm.embed(infile, message)

    #print 'Saving to', outfile

    #if n_rounds == 1:
    im_out.save(outfile, quality=args.quality)
    #im_out.show()
    #sys.exit()

    #print 'Loading from', outfile, 'to retrieve message'

    retrieved = wm.extract(outfile)

    if message != retrieved:
        if n_rounds == 1:
            print
            print 'Failure!'
            print message
            print retrieved
    else:
        if n_rounds == 1:
            print 'Success!'
        successes += 1

print successes, 'successful extractions out of', n_rounds

'''
print 'Getting the diff'
im_in = Image.open(infile)
in_bands = im_in.split()
out_bands = im_out.split()
diffs = [ImageMath.eval("convert(b-a, 'L')", a=in_bands[i], b=out_bands[i])
        for i in range(len(in_bands))]
diff = Image.merge('RGB', diffs)
diff.save('diff.png')
'''
