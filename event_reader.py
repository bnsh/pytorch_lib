#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""A tensorboard even file consists of a series of
    a unsigned long long (Q) (length of the "event_str")
    an unsigned int (I) a crc32 of the packed length above
    the actual event string (which must be length long)
    an unsigned int (I) a crc32 of the event str

   Arguably this doesn't belong in pytorchlib.. But, I wanted something that would
   read the events written to by tensorboardX.
"""

import os
import re
import struct
from tensorboard.compat.proto.event_pb2 import Event
from .crc32c import crc32c

# from https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/record_writer.py
def masked_crc32c(data):
    val = u32(crc32c(data))
    return u32(((val >> 15) | u32(val << 17)) + 0xa282ead8)

def u32(val):
    return val & 0xffffffff

def event_reader(filepointer):
    while True:
    # So, first read a Q
        eightbytes = filepointer.read(8)
        if len(eightbytes) != 8:
            break
        header, = struct.unpack("Q", eightbytes)
        fourbytes = filepointer.read(4)
        purported_header_crc32, = struct.unpack("I", fourbytes)
        data = filepointer.read(header)
        fourbytes = filepointer.read(4)
        purported_data_crc32, = struct.unpack("I", fourbytes)

        empiric_header_crc32 = masked_crc32c(struct.pack("Q", header))
        empiric_data_crc32 = masked_crc32c(data)

        assert empiric_header_crc32 == purported_header_crc32
        assert empiric_data_crc32 == purported_data_crc32

        evt = Event()
        evt.ParseFromString(data)
        yield evt

def last_epoch(dname):
    def traverse(dname, func, **kw):
        if os.path.isdir(dname):
            children = [os.path.join(dname, child) for child in sorted(os.listdir(dname))]
            for child in children:
                traverse(child, func, **kw)
        else:
            func(dname, **kw)

    max_epoch = [None]
    def update_max_epoch(fname):
        basename = os.path.basename(fname)
        match = re.match(r'^events.out.tfevents.[0-9]+.(.*)$', basename)
        if match is not None:
            with open(fname, "rb") as filep:
                for event in event_reader(filep):
                    if max_epoch[0] is None or (event.step is not None and event.step >= max_epoch[0]):
                        max_epoch[0] = event.step

    traverse(dname, update_max_epoch)
    return max_epoch[0]
