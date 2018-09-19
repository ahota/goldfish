import sys
sys.path.append('../')
from goldfish.entropy import EntropyWatermarker

length = 16

wm = EntropyWatermarker()
message = wm.extract(sys.argv[1],
        message_length=length*8)
print message
