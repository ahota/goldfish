import sys
sys.path.append('../')
from goldfish.energy import EnergyWatermarker
import uuid
import time
import os, binascii

from PIL import Image, ImageMath
from matplotlib import pyplot
from argparse import ArgumentParser

imagetypes = ('jpeg', 'png', 'bmp')
#channels = ('luma', 'cb', 'cr')
channels = ('R', 'G', 'B')

parser = ArgumentParser()
parser.add_argument('image', help='Path to the image to test with')
parser.add_argument('-n', '--n-rounds', type=int, default=1,
        help='Number of rounds to perform')
parser.add_argument('-t', '--type', choices=imagetypes, default=imagetypes[0],
        help='Filetype to save during test')
parser.add_argument('-b', '--bits-per-block', type=int, default=4,
        help='Bits per block to embed')
parser.add_argument('-c', '--channel', default=channels[0],
        help='Channel to embed in (luma is best)')
parser.add_argument('-q', '--quality', type=int, default=95,
        help='Quality factor to use when saving as JPEG')
parser.add_argument('--debug', action='store_true',
        help='Enable debug messages')
parser.add_argument('--super-debug', action='store_true',
        help='Enable debug messages and show embedding')
parser.add_argument('--quiet', action='store_true',
        help='Disable all output')
parser.add_argument('-d', '--data-size', type=int, default=32,
        help='How many bytes of watermark data to embed/extract')
parser.add_argument('--direct', action='store_true',
        help='Directly decode the watermarked image to test insertions/deletions')
args = parser.parse_args()

infile = sys.argv[1]
outfile = '.'.join(infile.split('.')[:-1])+'-altered.' + args.type
successes = 0

print_len = len(str(args.n_rounds))
if args.super_debug:
    args.debug = True

for i in range(args.n_rounds):
    if not args.quiet:
        print '{num:{w}}/{t}'.format(num=i+1, w=print_len, t=args.n_rounds)

    message = binascii.b2a_hex(os.urandom(max(1, args.data_size/2)))
    #message = uuid.uuid4().hex #format(uuid.uuid1().time_low, 'x')

    wm = EnergyWatermarker(bits=args.bits_per_block,
            chan=args.channel,
            debug=args.debug)

    im_out = wm.embed(infile, message)

    if args.super_debug:
        im = Image.open(infile)
        im.show()
        im_out.show()
        sys.exit()

    if not args.direct:
        im_out.save(outfile, format=args.type, quality=args.quality)
        im_out = Image.open(outfile)

    retrieved = wm.extract(im_out, message_length=args.data_size*8)

    if message != retrieved:
        if args.n_rounds == 1 and not args.quiet:
            print 'Failure!'
            print message
            print retrieved
    else:
        if args.n_rounds == 1 and not args.quiet:
            print 'Success!'
        successes += 1

if not args.quiet:
    print successes, 'out of', args.n_rounds

sys.exit(successes)

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
