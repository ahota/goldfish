import numpy, sys
sys.path.append('../')
from goldfish.entropy import EntropyWatermarker

from argparse import ArgumentParser
from matplotlib import pyplot
from PIL import Image

imagetypes = ('jpeg', 'png', 'bmp')

parser = ArgumentParser(description='Plot entropy of blocks in an image')
parser.add_argument('image', help='Path to the image to calulcate')
parser.add_argument('-t', '--type', choices=imagetypes, default=imagetypes[0],
        help='Filetype to save to in between calculations (default: {})'.format(
            imagetypes[0]))
parser.add_argument('-q', '--quality', type=int, default=95,
        help='Quality to save as when saving as jpeg (default: 95)')
parser.add_argument('--direct', action='store_true',
        help='Directly recalculate entropy instead of saving in between')
args = parser.parse_args()

wm = EntropyWatermarker(quality=args.quality)

merged, em1 = wm.entropy_heatmap(args.image, apply_qm=True)
print 'em1 min {:.4f} avg {:4f} max {:.4f}'.format(numpy.min(em1),
        numpy.average(em1), numpy.max(em1))

if not args.direct:
    outfile = '.'.join(args.image.split('.')[:-1])+'-heatmapped.' + args.type
    merged.save(outfile, format=args.type, quality=args.quality)
    merged = Image.open(outfile)

merged, em2 = wm.entropy_heatmap(merged)
print 'em2 min {:.4f} avg {:4f} max {:.4f}'.format(numpy.min(em2),
        numpy.average(em2), numpy.max(em2))

fig, axes = pyplot.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
fig.set_size_inches(13.3, 5)

hm = axes[0].imshow(em1, cmap='viridis', aspect=1)
axes[0].set(adjustable='box-forced', aspect='equal')
cbar = axes[0].figure.colorbar(hm, ax=axes[0])
cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

hm = axes[1].imshow(em2, cmap='viridis', vmax=numpy.max(em1), aspect=1)
axes[1].set(adjustable='box-forced', aspect='equal')
cbar = axes[1].figure.colorbar(hm, ax=axes[1])
cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

pyplot.show()
