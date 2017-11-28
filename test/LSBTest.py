import sys
sys.path.append('../')
from goldfish.watermarker import Watermarker
import uuid

from PIL import Image, ImageMath, ImageDraw, ImageFont
from argparse import ArgumentParser

fnt = ImageFont.load_default().font

channels = ('R', 'G', 'B')
imagetypes = ('bmp', 'jpeg', 'png')

parser = ArgumentParser()
parser.add_argument('image', help='Path to the image to test with')
parser.add_argument('-n', '--n-rounds', type=int, default=1,
        help='Number of rounds to perform')
parser.add_argument('-t', '--type', choices=imagetypes, default=imagetypes[0],
        help='Filetype to save during test')
parser.add_argument('-q', '--quality', type=int, default=75,
        help='Compression quality watermark should resist')
parser.add_argument('-k', '--bits-per-pixel', type=int, default=2,
        help='Bits per pixel to embed')
parser.add_argument('-c', '--channel', choices=channels, default=channels[0],
        help='Channel to embed in')
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

print_len = len(str(args.n_rounds))
if args.super_debug:
    args.debug = True
successes = 0

for i in range(args.n_rounds):
    if not args.quiet:
        print '{num:{w}}/{t}'.format(num=i+1, w=print_len, t=args.n_rounds)
    infile = sys.argv[1]
    outfile = infile[:-4]+'-altered.' + args.type

    message = uuid.uuid4().hex
    wm = Watermarker(bits_per_pixel=args.bits_per_pixel,
                     chan=args.channel,
                     debug=args.debug)

    if args.debug and not args.quiet:
        print 'Embedding message \"'+message+'\" into image'

    im = Image.open(infile)
    im_out = wm.embed(infile, message)

    if args.super_debug:
        im.show()
        im_out.show()
        sys.exit()

    # only reading back from image in memory
    # LSB doesn't survive most image compression
    if args.debug and not args.quiet:
        print 'Extracting'
    if args.direct:
        retrieved = wm.extract(im_out)
    else:
        im_out.save(outfile)
        retrieved = wm.extract(outfile)

    if message != retrieved:
        if args.debug and not args.quiet:
            print 'Failure!'
            print message
            print retrieved
    else:
        if args.debug and not args.quiet:
            print 'Success!'
        successes += 1

print successes, 'out of', args.n_rounds, 'successful'

'''
print 'Getting the diff'
im_in = Image.open(infile)
in_bands = im_in.split()
out_bands = im_out.split()
diff = ImageMath.eval("convert(b - a, 'L')", a=in_bands[0], b=out_bands[0])
diff.save('diff.png')
'''
