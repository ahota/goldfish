import sys
import os
import numpy
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, FixedLocator

from mpl_toolkits.axes_grid1 import ImageGrid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', type=str, nargs='+',
        help='input file from parameter_search.sh (e.g. d16.txt)')
parser.add_argument('-o', '--output-file', type=str,
        help='output filename (default: plot.png')
parser.add_argument('-n', '--name', type=str,
        help='title of the plot (default: <inputfile(s)>)')
parser.add_argument('-s', '--show', action='store_true', default=False,
        help='show the plot after saving it')
parser.add_argument('--only-show', action='store_true', default=False,
        help='only show the plot, and do not save it')

args = parser.parse_args()

fig = pyplot.figure(figsize=(16,3))
grid = ImageGrid(fig, 111,
        nrows_ncols=(1, len(args.inputfile)),
        axes_pad=0.25,
        share_all=True,
        cbar_location='right',
        cbar_mode='single',
        cbar_pad=0.25
        )
tables = {}

if args.name == None:
    args.name = ', '.join([os.path.basename(f) for f in args.inputfile])
if args.output_file == None:
    args.output_file = 'plot.png'

fig.suptitle(args.name)

xloc = FixedLocator([0, 9, 19, 29, 39, 49, 62])
yloc = FixedLocator([0, 4, 9, 14, 19, 24])
xfmt = FuncFormatter(lambda x, pos: int(x) + 1)
yfmt = FuncFormatter(lambda y, pos: int(y) * 10000 + 10000)

for fi, infile in enumerate(args.inputfile):
    dataset = infile.split('/')[1]
    if dataset == 'girus':
        dataset = 'giant virus'
    dataset = dataset.capitalize()
    tables[infile] = []
    with open(infile, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.split()
            tables[infile].append([int(e) for e in tokens[1:]])

    mat = numpy.array(tables[infile])
    grid[fi].set_title(dataset)
    grid[fi].xaxis.set_major_locator(xloc)
    grid[fi].xaxis.set_major_formatter(xfmt)
    grid[fi].yaxis.set_major_locator(yloc)
    grid[fi].yaxis.set_major_formatter(yfmt)
    im = grid[fi].imshow(mat, vmax=10, cmap='viridis')

grid[-1].cax.colorbar(im)
grid.cbar_axes[0].axis[grid.cbar_axes[0].orientation].label.set_text('# successful extractions')
#grid[-1].cax.set_yabel('# successful extractions')
grid[-1].cax.toggle_label(True)
grid[1].set_xlabel(r'Bits embedded per block $B$')
grid[0].set_ylabel(r'Entropy threshold $T$')


if not args.only_show:
    fig.savefig(args.output_file, bbox_inches='tight')

if args.show or args.only_show:
    pyplot.show()
