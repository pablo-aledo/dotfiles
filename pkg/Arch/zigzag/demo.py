# -*- coding: utf-8 -*-

from grammin import context
from gramutil import size_tesis
from libmisc import print_progress
from search import reorderer, zigzag

import sys

if len(sys.argv) < 2:
    sys.stderr.write("Argument missing: path to the file to apply ZZ.\n")
    sys.exit(1)

filename = sys.argv[1]
string = open(filename).read()

ctx = context(sequence=string)
r = reorderer(ctx)
print
print "Possible constituents:", len(ctx.cons)
print

print "Running zigzag"
zigzag(r, cback=print_progress(ctx.cons), sizef=size_tesis)
G = r.get_G()
print G
