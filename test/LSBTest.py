import sys
sys.path.append('../')
from goldfish.watermarker import Watermarker
import uuid

from PIL import Image, ImageMath, ImageDraw, ImageFont
from argparse import ArgumentParser
import os, random, time

fnt = ImageFont.load_default().font

channels = ('R', 'G', 'B')
imagetypes = ('bmp', 'jpeg', 'png')

parser = ArgumentParser()
parser.add_argument('image', help='Path to the image to test with')
parser.add_argument('-n', '--n-rounds', type=int, default=1,
        help='Number of rounds to perform')
parser.add_argument('-t', '--type', choices=imagetypes, default=imagetypes[0],
        help='Filetype to save during test')
parser.add_argument('-q', '--quality', type=int, default=95,
        help='Quality factor to use when saving as JPEG')
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
parser.add_argument('--stupid', action='store_true',
        help='Embed a JPG in the bitmap')
parser.add_argument('--time', action='store_true',
        help='Time the embedding process')
args = parser.parse_args()

print_len = len(str(args.n_rounds))
if args.super_debug:
    args.debug = True
successes = 0
times = []
ext_times = []
start = 0

for i in range(args.n_rounds):
    if not args.quiet:
        print '{num:{w}}/{t}'.format(num=i+1, w=print_len, t=args.n_rounds)
    infile = sys.argv[1]
    outfile = infile[:-4]+'-altered.' + args.type

    message = ''.join(random.choice('0123456789abcdef') for i in range(args.data_size))
    if args.type == 'bmp' and args.stupid:
        im = Image.open(infile)
        stupid_name = infile[:-4]+'-stupid.jpeg'
        im.save(stupid_name, quality=args.quality)
        message = open(stupid_name, 'rb').read()
    #message = uuid.uuid4().hex
    wm = Watermarker(bits_per_pixel=args.bits_per_pixel,
                     chan=args.channel,
                     debug=args.debug)

    if args.debug and not args.quiet:
        print 'Embedding message \"'+message+'\" into image'

    im = Image.open(infile)
    if args.time:
        start = time.time()
    im_out = wm.embed(infile, message)
    end = time.time()
    if args.time:
        times.append(end - start)

    if args.super_debug:
        im.show()
        im_out.show()
        sys.exit()

    # only reading back from image in memory
    # LSB doesn't survive most image compression
    if args.debug and not args.quiet:
        print 'Extracting'
    if args.direct:
        retrieved = wm.extract(im_out, message_length=args.data_size*8)
    else:
        im_out.save(outfile, quality=args.quality)
        if args.time:
            start = time.time()
        retrieved = wm.extract(outfile, message_length=args.data_size*8)
        end = time.time()
        if args.time:
            ext_times.append(end - start)

    if args.stupid:
        stupid_name = infile[:-4]+'-recovered.jpeg'
        f = open(stupid_name, 'wb')
        f.write(retrieved)
        f.close()

    if message != retrieved:
        if args.debug and not args.quiet:
            print 'Failure!'
            print message
            print retrieved
    else:
        if args.debug and not args.quiet:
            print 'Success!'
        successes += 1

if not args.quiet:
    print successes, 'out of', args.n_rounds, 'successful'
if args.time:
    print sum(times)/len(times), 's average embed time'
    print sum(ext_times)/len(ext_times), 's average extract time'

sys.exit(successes)

'''
print 'Getting the diff'
im_in = Image.open(infile)
in_bands = im_in.split()
out_bands = im_out.split()
diff = ImageMath.eval("convert(b - a, 'L')", a=in_bands[0], b=out_bands[0])
diff.save('diff.png')
'''
