import sys
sys.path.append('../')
from goldfish.energy import EnergyWatermarker
import uuid
import time

from PIL import Image, ImageMath

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
#outfile = '.'.join(infile.split('.')[:-1])+'-altered.png'
outfile = '.'.join(infile.split('.')[:-1])+'-altered.jpg'
n_rounds = 10
successes = 0

for i in range(n_rounds):
    if n_rounds != 1:
        sys.stdout.write('\r{}'.format(i))
        sys.stdout.flush()
    message = uuid.uuid4().hex #format(uuid.uuid1().time_low, 'x')

    wm = EnergyWatermarker()
    im_out = wm.embed(infile, message)
    if n_rounds == 1:
        im_out.show()

    if outfile.endswith('jpg'):
        im_out.save(outfile, quality=95)
    elif outfile.endswith('png'):
        im_out.save(outfile)

    retrieved = wm.extract(outfile)

    if message != retrieved:
        if n_rounds == 1:
            print 'Failure!'
            print message
            print retrieved
    else:
        if n_rounds == 1:
            print 'Success!'
        successes += 1

if n_rounds != 1:
    sys.stdout.write('\r{}\n'.format(i))
print successes, 'out of', n_rounds

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
