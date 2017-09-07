import sys
sys.path.append('../')
from goldfish.entropy import EntropyWatermarker
import uuid

from PIL import Image, ImageMath
from skimage import data

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

n_rounds = 100
successes = 0

print_len = len(str(n_rounds))

for i in range(n_rounds):
    sys.stdout.write('\r{num:{width}}/{total}'.format(num=i, width=print_len,
            total=n_rounds))
    sys.stdout.flush()
    infile = sys.argv[1]
    outfile = '.'.join(infile.split('.')[:-1])+'-altered.png'

    message = uuid.uuid4().hex
    wm = EntropyWatermarker()

    #print 'Embedding message \"'+message+'\" into image'

    im_out = wm.embed(infile, message)

    #print 'Saving to', outfile

    if n_rounds == 1:
        im_out.save(outfile)

    #print 'Loading from', outfile, 'to retrieve message'

    retrieved = wm.extract(im_out)

    if message != retrieved:
        if n_rounds == 1:
            print 'Failure!'
            print message
            print retrieved
    else:
        if n_rounds == 1:
            print 'Success!'
        successes += 1

sys.stdout.write('\r{}/{}\n'.format(n_rounds, n_rounds))
print successes, 'extractions out of', n_rounds

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
