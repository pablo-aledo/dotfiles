/*
 * Copyright 2015 Florian Benz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package oo

import "qsufsort"

type CompNt struct {
	Id     int
	Length int
}

type NonTerminal struct {
	Id     int
	Start  int
	Length int
	Count  int
}

// The graph used for occurrence optimization
type Graph [][]CompNt

// The main data structure for the internal representation of the given problem
type File struct {
	Name         string
	Data         []uint32
	Size         int
	Graph        Graph
	NonTerminals []*NonTerminal
	NtCount      int
}

type Individual struct {
	Score  int
	Choice []bool
}

type Solution struct {
	Score   int
	Choice  []bool
	NoScore bool
}

func NewFile(name string, data []uint32) *File {
	return &File{name, data, len(data), nil, nil, -1}
}

// Get all repeats (possible non-terminals) and construct a graph
// for occurrence optimization with all of them.
func (file *File) Construct() {
	tmpGraph := make(Graph, file.Size)

	sah := make([]int, file.Size)
	inv := make([]int, file.Size)
	ss := qsufsort.NewSuffixSortable(sah, inv, 1)
	sa := qsufsort.Qsufsort(ss, file.Data, 0)
	nts := make([]*NonTerminal, 0)
	ntsPtr := &nts

	// create suffix tree
	treeOccurrences := make([]int, 1)
	treeOccurrences[0] = sa[0]
	tree := &SuffixTree{
		StartIndex:  sa[0],
		EndIndex:    file.Size - 1,
		Occurrences: treeOccurrences,
		Children:    make([]*SuffixTree, 0),
	}
	for i := 1; i < len(sa); i++ {
		tree.Insert(file.Data, sa[i], sa[i])
	}

	// creates the graph
	tree.CreateGraph(tmpGraph, ntsPtr, 0)

	// Rebuild the 2d array so that it is not fragmented in the memory.
	// This plays an important role for the performance!
	graph := make(Graph, file.Size)
	for i, pos := range tmpGraph {
		if pos != nil {
			graph[i] = make([]CompNt, len(pos))
			for j, nt := range pos {
				graph[i][j].Id = nt.Id
				graph[i][j].Length = nt.Length
			}
		} else {
			graph[i] = make([]CompNt, 0)
		}
	}

	file.Graph = graph
	file.NonTerminals = *ntsPtr
	file.NtCount = len(file.NonTerminals)
}
