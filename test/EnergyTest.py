import sys
sys.path.append('../')
from goldfish.energy import EnergyWatermarker
import uuid
import time

from PIL import Image, ImageMath
from matplotlib import pyplot

if len(sys.argv) < 2:
    print 'Usage:'
    print '\t', sys.argv[0], 'image_filepath'
    sys.exit(1)

infile = sys.argv[1]
outfile = '.'.join(infile.split('.')[:-1])+'-altered.png'
#outfile = '.'.join(infile.split('.')[:-1])+'-altered.jpg'
n_rounds = 1
successes = 0

# timing stuff
total_time = 0
times = []

for i in range(n_rounds):
    if n_rounds != 1:
        sys.stdout.write('\r{}'.format(i))
        sys.stdout.flush()
    message = uuid.uuid4().hex #format(uuid.uuid1().time_low, 'x')

    wm = EnergyWatermarker(debug=False)
    start = time.clock()
    im_out = wm.embed(infile, message)
    end = time.clock()
    total_time += end - start
    times.append((end - start)*1000)
    if n_rounds == 1:
        #im_out.show()
        pass

    if outfile.endswith('jpg'):
        im_out.save(outfile, quality=95)
    elif outfile.endswith('png'):
        im_out.save(outfile)

    retrieved = wm.extract(im_out) #outfile)

    if message != retrieved:
        if n_rounds == 1:
            print 'Failure!'
            print message
            print retrieved
    else:
        if n_rounds == 1:
            print 'Success!'
            Image.open(outfile).show()
        successes += 1

if n_rounds != 1:
    sys.stdout.write('\r{}\n'.format(i))
print successes, 'out of', n_rounds

print 'Average embedding time:', 1000*total_time/n_rounds, 'ms'
#pyplot.hist(times)
#pyplot.show()

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
