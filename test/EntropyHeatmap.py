import numpy, random, sys
sys.path.append('../')
from goldfish.entropy import EntropyWatermarker

from argparse import ArgumentParser
from matplotlib import pyplot
from matplotlib import colors
from PIL import Image

imagetypes = ('jpeg', 'png', 'bmp')

parser = ArgumentParser(description='Plot entropy of blocks in an image')
parser.add_argument('image', help='Path to the image to calulcate')
parser.add_argument('-t', '--type', choices=imagetypes, default=imagetypes[0],
        help='Filetype to save to in between calculations (default: {})'.format(
            imagetypes[0]))
parser.add_argument('-b', '--bits-per-block', type=int, default=16,
        help='Bits per block to embed')
parser.add_argument('-d', '--data-size', type=int, default=32,
        help='How many bytes of watermark data to embed/extract')
parser.add_argument('-q', '--quality', type=int, default=95,
        help='Quality to save as when saving as jpeg (default: 95)')
parser.add_argument('-e', '--entropy-threshold', type=int, default=4000,
        help='Block entropy threshold')
parser.add_argument('--before-after', action='store_true',
        help='Show heatmap of entropy change during embedding')
parser.add_argument('--direct', action='store_true',
        help='Directly recalculate entropy instead of saving in between')
args = parser.parse_args()

# from: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = pyplot.cm.get_cmap(base_cmap)
    color_list = base(numpy.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def entropy_before_after(args):
    wm = EntropyWatermarker(quality=args.quality,
            threshold=args.entropy_threshold,
            bits=args.bits_per_block)

    hex_digits = '0123456789abcdef'
    message = ''.join(random.choice(hex_digits) for _ in range(args.data_size))
    statuses = wm.check_embed_entropy(args.image, message)
    status_names = ['eom', 'below threshold', 'below after embedding', 'above after embedding']

    fig, ax = pyplot.subplots()
    fig.set_size_inches(5, 5)

    hm = pyplot.imshow(statuses, cmap=discrete_cmap(4, 'viridis'), aspect=1)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.set_title('entropy change during embedding for e={}'.format(args.entropy_threshold))
    formatter = pyplot.FuncFormatter(lambda val, loc: status_names[loc])
    cbar = ax.figure.colorbar(hm, ax=ax, ticks=range(4), format=formatter)

    pyplot.show()


def entropy_heatmap(args):
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
    axes[0].set_title('initial entropy')
    cbar = axes[0].figure.colorbar(hm, ax=axes[0])
    cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

    hm = axes[1].imshow(em2, cmap='viridis', vmax=numpy.max(em1), aspect=1)
    axes[1].set(adjustable='box-forced', aspect='equal')
    if args.direct:
        axes[1].set_title('entropy after merging')
    else:
        axes[1].set_title('entropy after merging and saving to '+args.type)
    cbar = axes[1].figure.colorbar(hm, ax=axes[1])
    cbar.ax.set_ylabel('entropy', rotation=-90, va='bottom')

    pyplot.show()

if args.before_after:
    entropy_before_after(args)
else:
    entropy_heatmap(args)
