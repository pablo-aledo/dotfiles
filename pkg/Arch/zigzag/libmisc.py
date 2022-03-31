# -*- coding: utf-8 -*-

import sys
import time


def xi(a):
    """
    Useful to iterate over the indexes of and object a
    """
    return xrange(len(a))


class str_alphacode:
    def __init__(self, string=""):
        self.ac = [None for x in range(256)]  # All chars
        self.adc = []
        self.rw = self.rewrite(string)

    def code(self, c):
        i = ord(c)  # Not sure how it will work with unicode
        if self.ac[i] is None:
            self.ac[i] = len(self.adc)
            self.adc.append(c)
        return self.ac[i]

    def cont_decode(self, i):
        return self.decode(i)

    def slice_decode(self, i, j):
        s = ""
        for k in xrange(i, j):
            s += self.decode(self.rw[k])
        return s

    def decode(self, i):
        return self.adc[i]

    def get_rewrite(self):
        return self.rw

    def rewrite(self, s):
        rw = []
        for c in s:
            cod = self.code(c)
            rw.append(cod)
        return rw

    def alphalen(self):
        return len(self.adc)

    def __len__(self):
        return self.alphalen()

    def __contains__(self, x):
        i = ord(x)
        return self.ac[i] is not None


class generic_alphacode(str_alphacode):
    def __init__(self, string=[]):
        self.ac = {}
        self.adc = []
        self.rw = self.rewrite(string)

    def code(self, c):
        if not c in self.ac:
            self.ac[c] = len(self.adc)
            self.adc.append(c)
        return self.ac[c]

    def cont_decode(self, i):
        return [self.decode(i)]

    def slice_decode(self, i, j):
        s = []
        for k in xrange(i, j):
            s.append(self.decode(self.rw[k]))
        return s

    def __contains__(self, x):
        return x in self.ac


def alphacode(string=[]):
    if isinstance(string, str):
        return str_alphacode(string)
    return generic_alphacode(string)


class print_progress(object):
    def __init__(self, cons):
        self.cons = cons
        self.lasttime = time.time()

    def __call__(self, action, consid, size):
        if action == "add":
            actionstr = "Adding\t"
        elif action == "rm":
            actionstr = "Removing"
        else:
            assert False
        newtime = time.time()
        print "%s\t%s (plain: %s)\t%i\ttime delta= %f s" % (actionstr,
            str(self.cons.get_ij(consid)),
            repr(self.cons.get_plain(consid)),
            size,
            newtime - self.lasttime)
        self.lasttime = newtime
        sys.stdout.flush()
