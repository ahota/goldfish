import numpy
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter

table = []

with open('spec_thresh_table.txt') as f:
    for line in f.readlines():
        table.append([int(e) for e in line.split()])

mat = numpy.array(table)
fig, ax = pyplot.subplots()
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(1000 + 100*x)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: y/10.0))
pyplot.ylabel('specular amount')
pyplot.xlabel('entropy threshold')
pyplot.imshow(mat, cmap='viridis')
pyplot.show()
