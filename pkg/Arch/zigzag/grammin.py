# -*- coding: utf-8 -*-

from libmisc import *


class constituents(object):
    def __init__(self, x):
        if isinstance(x, str_alphacode):
            self.from_alphac(x)
        elif isiterable(x):
            self.from_secuence(x)
        else:
            raise Exception("must be initialized with iterable or "
                            "an alphacode-compatible class")

    def from_secuence(self, secuence):
        self.from_alphac(alphacode(secuence))

    def from_alphac(self, alphac):
        # Seeding base case, strings of len >= 2
        self.ac = alphac
        string = self.ac.get_rewrite()
        alphan = self.ac.alphalen()

        last = [None for i in xrange(alphan * alphan)]
        freq = [0 for i in xrange(alphan * alphan)]
        strs = [[] for i in xrange(alphan * alphan)]
        for i in xrange(len(string) - 1):
            a = string[i]
            b = string[i + 1]
            index = a * alphan + b
            lista = strs[index]
            lista.append(i)
            if len(lista) <= 1 or lista[-1] - last[index] >= 2:
                freq[index] += 1
                last[index] = i
        i = 0
        longs = []
        #filtrando para freq >= 2
        while i < len(strs):
            if freq[i] < 2:
                strs[i] = strs[-1]
                freq[i] = freq[-1]
                strs.pop()
                freq.pop()
                last.pop()
            else:
                i += 1
                longs.append(2)

        newstrs = [[] for i in xrange(alphan)]
        newfreq = [0 for i in xrange(alphan)]
        last = [None for i in xrange(alphan)]
        k = 0
        while k != len(strs):
            curr = strs[k]
            lon = longs[k]
            for start in curr:
                end = start + lon
                if end < len(string):
                    index = string[end]
                    lista = newstrs[index]
                    lista.append(start)
                    if len(lista) == 1 or lista[-1] - last[index] >= lon + 1:
                        newfreq[index] += 1
                        last[index] = start
            for i in xrange(alphan):
                if newfreq[i] < 2:
                    while len(newstrs[i]) != 0:
                        newstrs[i].pop()
                else:
                    strs.append(newstrs[i])
                    freq.append(newfreq[i])
                    longs.append(lon + 1)
                    newstrs[i] = []
                newfreq[i] = 0
                last[i] = None
            k += 1
        self.strs = strs
        self.longs = longs
        self.freq = freq
        self.plain = [None for x in self.longs]

    def __len__(self):
        return len(self.strs)

    def __getitem__(self, i):
        os = self.strs[i]
        l = self.longs[i]
        for j in xrange(len(os)):
            o = os[j]
            yield o, o + l

    def get_ij(self, k):
        i = min(self.strs[k])
        return i, i + self.longs[k]

    def get_alphac(self):
        return self.ac

    def get_len(self, i):
        return self.longs[i]

    def get_non_overlap_freq(self, i):
        return self.freq[i]

    def get_overlap_freq(self, i):
        return len(self.strs[i])

    def get_index_from_ij(self, i, j):
        ijl = j - i
        for k in xrange(len(self.strs)):
            l = self.longs[k]
            if ijl != l:
                continue
            for start in self.strs[k]:
                if start == i:
                    return k
        return None

    def __getstate__(self):
        return (self.ac, self.strs, self.longs)

    def __setstate__(self, s):
        self.ac, self.strs, self.longs = s
        self.plain = [None for x in self.longs]
        self.freq = []
        for k in xrange(len(self.strs)):
            l = self.longs[k]
            ocurrs = self.strs[k]
            last = ocurrs[0]
            freq = 1
            for i in ocurrs:
                if i >= last + l:
                    last = i
                    freq += 1
            self.freq.append(freq)

    def get_index(self, s):
        i = 0
        while i != len(self) and self.get_plain(i) != s:
            i += 1
        if i == len(self):
            return None
        return i

    def get_plain(self, k):
        if self.plain[k] is None:
            self.plain[k] = self._get_plain(k)
        return self.plain[k]

    def get_plain_constituents(self):
        for i in xrange(len(self.plain)):
            if self.plain[i] is None:
                self.get_plain(i)
        return self.plain

    def _get_plain(self, k):
        i, j = self.get_ij(k)
        return self.ac.slice_decode(i, j)


def rule_body(alphac=None):
    if alphac is None:
        return generic_body()
    else:
        return slim_body(alphac)


class base_body(object):
    def __init__(self):
        self.body = []

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

    def clear(self):
        self.body = []

    def __str__(self):
        showf = lambda x: str(x)
        try:
            showf = self.alphac.decode
        except AttributeError:
            pass
        s = ""
        for nt, e in self:
            if nt:
                s += "@" + str(e) + " "
            else:
                s += showf(e) + " "
        return s[:-1]


class generic_body(base_body):
    """
    Una clase de conveniencia usada para pasar enviar y recibir
    reglas en tree_grammar.
    """
    def __getitem__(self, i):
        return self.body[i]

    def append(self, nont, elem):
        self.body.append((nont, elem))


class slim_body(base_body):
    """
    Una clase de conveniencia usada para pasar enviar y recibir
    reglas en tree_grammar.
    """
    def __init__(self, alphac):
        base_body.__init__(self)
        self.alphac = alphac

    def __getitem__(self, i):
        nont = False
        e = self.body[i]
        if e >= len(self.alphac):
            e = e - len(self.alphac)
            nont = True
        return nont, e

    def append(self, nont, elem):
        if nont:
            elem += len(self.alphac)
        self.body.append(elem)


class tree_grammar(object):
    def __init__(self):
        self.clear()

    def add_body(self, rule):
        i = len(self.rules)
        self.rules.append(rule)
        return i

    def set_S(self, i):
        self.S = i

    def get_S(self):
        return self.S

    def __str__(self):
        s = ""
        for i in xi(self.rules):
            s += "%i -> %s\n" % (i, str(self.rules[i]))
        return s

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, i):
        return self.rules[i]

    def iter(self):
        return iter(self.rules)

    def clear(self):
        self.rules = []
        self.S = 0


class context(object):
    """
    A convenience class to keep together stuff that's always needed.
    """
    def __init__(self, sequence=None, alphabet=None, cons=None, grammar=None):
        self.sequence = sequence
        self.alphabet = alphabet
        self.cons = cons
        self.grammar = grammar
        self.fill()

    def fill_grammar(self):
        if self.grammar is None:
            self.grammar = tree_grammar()

    def fill_cons(self):
        if self.cons is None:
            self.fill_alphabet()
            self.cons = constituents(self.alphabet)

    def fill_alphabet(self):
        if self.alphabet is None:
            self.fill_sequence()
            self.alphabet = alphacode(self.sequence)

    def fill_sequence(self):
        if self.sequence is None:
            assert self.grammar is not None
            self.sequence = get_yields(self.grammar)[self.grammar.get_S()]

    def fill(self):
        assert self.sequence is not None or self.grammar is not None
        self.fill_grammar()
        self.fill_cons()
        self.fill_alphabet()
        self.fill_sequence()
