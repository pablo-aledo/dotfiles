# -*- coding: utf-8 -*-

"""
This file contains stuff useful to work with tree_grammar.
It divided into:
    - Parsing
    - Iteration over tree_grammar elements
    - Frequency, entropy and scores
"""

from libmisc import xi, generic_alphacode
from grammin import rule_body, tree_grammar

###               ###
###    PARSING    ###
###               ###


def parse_body(s, nont_reader=lambda x: x, elem_reader=lambda x: x):
    body = rule_body()
    s = s.split()
    for e in s:
        if e[0] == '@':
            code = nont_reader(int(e[1:]))
            body.append(True, code)
        else:
            code = elem_reader(e)
            body.append(False, code)
    return body


def parse_grammar(s, alphac):
    # The "trick" in this implementation is that alphacode assigns increasing
    # values from 0 and tree_grammar too. This is postcondition of both.
    lines = s.split("\n")
    nont_alpha = generic_alphacode()
    G = tree_grammar()
    rules = {}
    for line in lines:
        if line == "":
            continue
        nt, body = line.split(" -> ")
        nt = int(nt)
        nont_alpha.code(nt)
        body = parse_body(body, nont_alpha.code, alphac.code)
        rules[nt] = body
    assert len(rules) == len(nont_alpha)
    for i in xi(rules):
        nt = nont_alpha.decode(i)
        j = G.add_body(rules[nt])
        assert i == j
    S = search_S(G)
    assert S is not None
    G.set_S(S)
    return G

###                 ###
###    ITERATION    ###
###                 ###


def iterate_in_codification_order(G, root=None, end_rule_symbol=None):
    if root is None:
        root = G.get_S()
    done = [False for i in xi(G)]
    done[root] = True
    q = [(root, 0)]
    while len(q) != 0:
        current, pos = q.pop()
        body = G[current]
        i = pos
        while i < len(body):
            nont, e = body[i]
            yield nont, e
            if nont and not done[e]:
                q.append((current, i + 1))
                q.append((e, 0))
                break
            i += 1
        if i == len(body):
            done[current] = True
            if current != root:
                yield True, end_rule_symbol


def iterate_tree_dfs(G, root=None):
    if root is None:
        root = G.get_S()
    q = [(root, 0)]
    yield True, root
    while len(q) != 0:
        current, i = q.pop()
        body = G[current]
        while i < len(body):
            nont, e = body[i]
            yield nont, e
            i += 1
            if nont:
                q.append((current, i))
                q.append((e, 0))
                break


def get_a_bracket_for_nonterminals(G):
    start = [None for i in xi(G)]
    end = [None for i in xi(G)]
    root = G.get_S()
    start[root] = 0
    globalpos = 0
    q = [root]
    for nont, e in iterate_in_codification_order(G, root):
        if nont:
            if e is None:
                end[q.pop()] = globalpos
            elif end[e] is not None:
                globalpos += end[e] - start[e]
            else:
                q.append(e)
                start[e] = globalpos
        else:
            globalpos += 1
    end[root] = globalpos
    ijs = []
    for i in xi(start):
        ijs.append((start[i], end[i]))
    return ijs


def iterate_brackets(G, root=None):
    if root is None:
        root = G.get_S()
    ls = [j - i for i, j in get_a_bracket_for_nonterminals(G)]
    i = 0
    for nont, e in iterate_tree_dfs(G, root):
        if nont:
            yield i, i + ls[e]
        else:
            i += 1


def get_yields(G):
    ys = [None for i in xi(G)]
    root = G.get_S()
    ys[root] = []
    q = [root]
    for nont, e in iterate_in_codification_order(G):
        if nont:
            if e is None:
                last = q.pop()
                ys[q[-1]].extend(ys[last])
            elif ys[e] is not None:
                ys[q[-1]].extend(ys[e])
            else:
                q.append(e)
                ys[e] = []
        else:
            ys[q[-1]].append(e)
    return ys


def nont_freq(G):
    freq = [0 for i in xi(G)]
    for i in xi(G):
        for nont, e in G[i]:
            if nont:
                freq[e] += 1
    return freq


def search_S(G):
    freq = nont_freq(G)
    for i in xi(freq):
        if freq[i] == 0:
            return i
    return None


###               ###
###    METRICS    ###
###               ###


def size_tesis(G):
    i = len(G) - 1
    for body in G:
        i += len(body)
    return i


def size_lehman_shelat(G):
    i = 0
    for body in G:
        i += len(body)
    return i
