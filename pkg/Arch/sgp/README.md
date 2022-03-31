Heuristics for the Smallest Grammar Problem (SGP)
==========

The smallest grammar problem is the problem of finding the smallest context-free grammar that generates exactly one given sequence. Approximating the problem with a ratio of less than 8569/8568 is known to be NP-hard. Most work on this problem has focused on finding decent solutions fast (mostly in linear time), rather than on good heuristic algorithms.

This project provides a hybrid of a max-min ant system and a genetic algorithm that in combination with a novel local search outperforms the state of the art on all files of the Canterbury corpus, a standard benchmark suite. Furthermore, this hybrid performs well on a standard DNA corpus.

Requirements
---------
* Go version 1.x

License
---------

Most of the code is released under the Apache 2.0 license (see LICENSE). The only exception is src/qsufsort/qsufsort.go which is a modified version of the file from the Go library and thus released under a BSD-style license (see src/qsufsort/LICENSE). All source code files contain a copyright notice header.

Usage
-------------

To build everything, use:

    build.sh

A explanation of all command line arguments is provided with:

    sgp --help

```
Usage of sgp:
  -choice="": file containing an encoded choice of yields to start with
  -cpuprofile="": enables CPU profiling
  -csv=false: output in CSV format
  -i="": input file
  -initls=false: save intermediate results of initial local search
  -irr=10: number of IRR runs
  -ls=true: use local search
  -lscandidates=1000: max #candidates considered for bottom-up local search
  -m="hybrid": metaheuristic (genetic, ants, hybrid, none)
  -number=-1: this number is appended to the created files
  -outputlimit=-1: output every grammar of this or smaller size
  -print=false: prints the grammar human readable to stdout
  -rirr=false: randomized IRR
  -seed=-1: seed for pseudo-random number generator
  -sizelimit=-1: stop if a grammar of this size is reached
  -timelimit=0: time limit with unit (e.g. 10s)
```

Run the hybrid

    sgp --i <input file> --timelimit 10s

Any result can easily be checked for correctness with a separate small checker. The checker is an independent programm that has less than 200 lines of code. Thus, the implementation fulfills the requirements of a [Certifying Algorithm](http://people.mpi-inf.mpg.de/~mehlhorn/ftp/CertifyingAlgorithms.pdf). The choice of nonterminals is used as a compact way of storing grammars and also plays the role of the witness:

    checker --f <input file> --g <grammar to verify> -s <grammar size>

Resources
-------------

Canterbury corpus:  http://corpus.canterbury.ac.nz/

Background
-------------

This project started during my Master's thesis at Saarland University in winter 2012/13. The results are published in the following paper:

[An Effective Heuristic for the Smallest Grammar Problem](http://dl.acm.org/citation.cfm?id=2463441). Florian Benz and Timo Kötzing. GECCO '13: Proceedings of the 15th International Conference on Genetic and Evolutionary Computation 
