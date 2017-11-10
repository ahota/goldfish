# new message structure:
# Assuming 1024 bits long
# Fingerprint can be 32 bits, like the PIC in Bo Li's method
# Remaining 992 bits (124 bytes) can be split into
# 10 12-byte fields and 1 4-byte field
# OR
# 9 12-byte fields and 1 16-byte field
#
#   0 -   31 : 32-bit fingerprint for detection
#  32 -  159 : 128-bit (16-byte) field for author name
# 160 -  255 : 96-bit (12-byte) field for organization?
# 256 -  351 : 96-bit (12-byte) field for dataset name
# 352 -  447 : 96-bit (12-byte) field for variable name
# 448 -  543 : 96-bit (12-byte) field for time step
# 544 -  639 : 96-bit (12-byte) field for type (e.g. simulation, observation)
# 640 -  735 : 96-bit (12-byte) field for project name?
# 736 -  831 : 96-bit (12-byte) field for ?
# 832 -  927 : 96-bit (12-byte) field for ?
# 928 - 1023 : 96-bit (12-byte) field for more information key

import random
import uuid

from collections import OrderedDict

def generate_key():
    return uuid.uuid4().hex

def ascii2bin(asciistr):
    return ''.join([format(ord(c), 'b').zfill(8) for c in asciistr])

def bin2ascii(binstr):
    return ''.join([chr(int(binstr[i:i+8], 2))
        for i in range(0, len(binstr), 8)])

goldfish_watermark_fingerprint = '11110000101000110000111101011100'
# lengths in bytes
goldfish_message_component_lengths = [4, 16] + [12]*9
goldfish_message_components = OrderedDict()
goldfish_message_components['fingerprint'] = [bin2ascii(goldfish_watermark_fingerprint)]
goldfish_message_components['author'] = ['ahota', 'cq1782', 'mahmadza', 'thobson2', 'kdawes', 'huangj', 'nbogda']
goldfish_message_components['organization'] = ['seelab', 'ornl', 'curent', 'nics', 'mabe', 'cisml', 'nimbios']
goldfish_message_components['dataset'] = ['supernova', 'magnetic', 'tornado', 'turbulence', 'teapot']
goldfish_message_components['variable'] = ['wind velocity magnitude', 'vorticity', 'mixfrac', 'tea', 'angular velocity']
goldfish_message_components['timestep'] = range(10)
goldfish_message_components['type'] = ['simulation', 'observation', 'interpolated', 'aggregation']
goldfish_message_components['project'] = ['tapestry', 'ecamp', 'watermarking', 'holotapestry', 'ipcc', 'vtkm']
goldfish_message_components['unknown1'] = ['tbd']
goldfish_message_components['unknown2'] = ['tbd']
goldfish_message_components['moreinformation'] = [generate_key()]*4

def create_dummy_message():
    message = ''
    # for each message component, choose a field value
    # limit it or pad it to the length of that field
    for ki, k in enumerate(goldfish_message_components):
        field = str(random.choice(goldfish_message_components[k]))
        binfield = ascii2bin(field[:goldfish_message_component_lengths[ki]])
        binfield += '0'*(goldfish_message_component_lengths[ki]*8-len(binfield))
        message += binfield
    return bin2ascii(message)

if __name__ == '__main__':
    m = create_dummy_message()
    print 'Created dummy message:', m
    print 'Message length:', len(m), 'bits'
    b = bin2ascii(m)
    print 'ASCIIfied message:', b
    # test usernames are at most 8 bytes
    # fingerprint is 4 bytes
    # so 13th byte should always be \0
    print '13th byte:', m[13*8:14*8]
