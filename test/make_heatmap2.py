import sys
import os
import numpy
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter

table = []

with open(sys.argv[1], 'r') as f:
    for line in f.readlines()[1:]:
        table.append([int(e) for e in line.split()[1:]])

pref = os.path.basename(sys.argv[1]).split('.')[0] 

mat = numpy.array(table)
fig, ax = pyplot.subplots()
ax.set_title(pref)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: int(y*250 + 250)))
pyplot.xlabel('bits per pixel')
pyplot.ylabel('entropy threshold')
pyplot.imshow(mat, cmap='viridis')

ofname = '/home/ahota/Pictures/' + pref + '_heatmap.png'
fig.savefig(ofname, bbox_inches='tight')
pyplot.show()
