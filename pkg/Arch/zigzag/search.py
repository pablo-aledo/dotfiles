# -*- coding: utf-8 -*-

from libmisc import xi
from grammin import rule_body
from gramutil import size_tesis


class reordering_graph(object):
    def __init__(self, seqlen):
        """
        Builds a reordering graph for a sequence of length seqlen
        """
        self.prev = [[] for i in xrange(seqlen + 1)]
        self.psym = [[] for i in xrange(seqlen + 1)]

    def clear(self):
        N = len(self)
        self.prev = [[] for i in xrange(N)]
        self.psym = [[] for i in xrange(N)]

    def clear_to_1(self):
        for i in xi(self.prev):
            del self.prev[i][1:]
            del self.psym[i][1:]

    def add(self, i, j, symbol):
        """
        Adds a edge between node i and node j labeled with symbol
        """
        self.prev[j].append(i)
        self.psym[j].append(symbol)

    def pop(self, i, j):
        assert i == self.prev[j].pop()
        self.psym[j].pop()

    def __getitem__(self, x):
        """
        Iterate over the edges arriving to x.
        Returns pairs (prev_index,symbol_label) that represent
        and edge between prev_index and x with label symbol label
        """
        for i in xi(self.prev[x]):
            yield self.prev[x][i], self.psym[x][i]

    def prevs(self, x):
        """
        Iterates ever the indexes (y) of all the edges of the form
        y -> x with label L
        The order of the iteration matches the order of psysms(x)
        """
        return self.prev[x]

    def psyms(self, x):
        """
        Iterates ever the labels (L) of all the edges of the form
        y -> x with label L
        The order of the iteration matches the order of prevs(x)
        """
        return self.psym[x]

    def __len__(self):
        return len(self.prev)


class reordering_paths(object):
    def __init__(self, graph, i, j):
        self.g = graph
        self.i = i
        self.j = j
        N = j - i + 1
        self.dist = [i for i in xrange(N)]
        self.ties = [[] for i in xrange(N)]
        self.sym = [[] for i in xrange(N)]

    def mindist(self, i=-1, j=-1):
        if i == -1:
            i = self.i
        if j == -1:
            j = self.j
        i -= self.i
        j -= self.i
        return self.dist[j] - self.dist[i]

    def get_ij(self):
        return self.i, self.j

    def __len__(self):
        return self.mindist()

    def calculate_dists(self, i=-1, j=-1):
        if i == -1:
            i = self.i
        if j == -1:
            j = self.j

        for x in xi(self.dist):
            self.dist[x] = x + 1
        self.dist[0] = 0

        for x in xrange(i, j + 1):
            for prev in self.g.prevs(x):
                if prev < i:
                    continue
                self.dist[x - self.i] = min(self.dist[x - self.i],
                                            self.dist[prev - self.i])

    def calculate_paths(self, i=-1, j=-1):
        if i == -1:
            i = self.i
        if j == -1:
            j = self.j

        for x in xi(self.dist):
            self.dist[x] = x + 1
        self.dist[0] = 0

        for x in xrange(i, j + 1):
            for prev, sym in self.g[x]:
                if prev < i:
                    continue
                if prev == self.i and x == self.j:
                    continue
                dx = self.dist[x - self.i]
                dp = self.dist[prev - self.i]
                if dx < dp + 1:
                    continue
                elif dx == dp + 1:
                    self.ties[x - self.i].append(prev - self.i)
                    self.sym[x - self.i].append(sym)
                elif dx > dp + 1:
                    del self.ties[x - self.i][:]
                    self.ties[x - self.i].append(prev - self.i)
                    self.dist[x - self.i] = dp + 1
                    del self.sym[x - self.i][:]
                    self.sym[x - self.i].append(sym)

    def get_path(self):
        return path(self.ties, self.sym, self.dist)

    def __iter__(self):
        return iter(self.get_path())


class path(object):
    def __init__(self, ties, sym, dist):
        self.dist = dist
        self.ties = ties
        self.sym = sym

    def __len__(self):
        return self.dist[-1]

    def __iter__(self):
        qsym = []
        i = len(self.dist) - 1
        while i != 0:
            qsym.append(self.sym[i][0])
            i = self.ties[i][0]
        qsym.reverse()
        for s in qsym:
            yield s


class reorderer(object):
    """
    A convenience class for efficient search of smallest grammars within
    the Minimal Grammar Parsing paradigm.

    You can "push" constituents, "pop" them out and "minimize" the target
    string by building a smallest grammar for que included constituents.
    """
    def __init__(self, context):
        self.ctx = context
        self.G = None
        self.q = []
        rw = self.ctx.alphabet.get_rewrite()
        self.graph = reordering_graph(len(rw))
        for i in xi(rw):
            self.graph.add(i, i + 1, rw[i])
        self.paths = [reordering_paths(self.graph, 0, len(rw))]
        self.minimize()

    def get_context(self):
        return self.ctx

    def clear(self):
        self.G = None
        self.q = []
        self.graph.clear_to_1()
        self.paths = self.paths[:1]

    def push(self, c):
        self.q.append(c)
        for i, j in self.ctx.cons[c]:
            self.graph.add(i, j, -len(self.paths))
        self.paths.append(reordering_paths(self.graph, i, j))

    def pop(self):
        c = self.q.pop()
        for i, j in self.ctx.cons[c]:
            self.graph.pop(i, j)
        self.paths.pop()

    def load(self, cs):
        self.clear()
        for c in cs:
            self.push(c)

    def minimize(self):
        for path in self.paths:
            path.calculate_paths()

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def get_q(self):
        return self.q

    def get_G(self):
        G = self.ctx.grammar
        G.clear()
        for path in self.paths:
            p = path.get_path()
            body = rule_body(self.ctx.alphabet)
            for s in p:
                if s < 0:
                    body.append(True, -s)
                else:
                    body.append(False, s)
            G.add_body(body)
        G.set_S(0)
        return G


def skip(action, consid, size):
    pass


def step_bottomup(R, sizef=size_tesis):
    cons = R.get_context().cons
    R.minimize()
    size = sizef(R)
    q = set(R.get_q())
    answer = None
    for c in xi(cons):
        if c in q:
            continue
        R.push(c)
        R.minimize()
        newsize = sizef(R)
        R.pop()
        if newsize <= size:
            size = newsize
            answer = c
    return answer, size


def bottomup(R, sizef=size_tesis, cback=skip):
    next, size = step_bottomup(R, sizef)
    while next is not None:
        cback("add", next, size)
        R.push(next)
        next, size = step_bottomup(R, sizef)
    R.minimize()


def step_topdown(R, sizef=size_tesis):
    R.minimize()
    size = sizef(R)
    q = R.get_q()
    answer = None
    for c in q:
        j = q.index(c)
        newq = q[:j] + q[j + 1:]
        R.load(newq)
        R.minimize()
        newsize = sizef(R)
        R.push(c)
        if newsize <= size:
            size = newsize
            answer = c
    return answer, size


def topdown(R, sizef=size_tesis, cback=skip):
    next, size = step_topdown(R, sizef)
    while next is not None:
        cback("rm", next, size)
        q = R.get_q()
        j = q.index(next)
        q = q[:j] + q[j + 1:]
        R.load(q)
        next, size = step_topdown(R, sizef)
    R.minimize()


def zigzag(R, sizef=size_tesis, cback=skip):
    """
    R must be an instance of the reorderer class.

    """
    R.minimize()
    newsize = sizef(R)
    size = newsize + 1
    while newsize < size:
        bottomup(R, sizef, cback)
        topdown(R, sizef, cback)
        size = newsize
        newsize = sizef(R)
