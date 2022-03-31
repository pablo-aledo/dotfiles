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

type SuffixTree struct {
	StartIndex  int
	EndIndex    int
	Occurrences []int
	Children    []*SuffixTree
}

// Checks wheter the node starts with the given character
func (t *SuffixTree) StartsWith(data []uint32) uint32 {
	if t.StartIndex > t.EndIndex {
		panic("start index is greater than the end index")
	}
	return data[t.StartIndex]
}

// Inserts a new suffix into the tree. This is done in a compressed way and
// means that only one split is done per insertion.
func (t *SuffixTree) Insert(data []uint32, startIndex, currentIndex int) {
	length := t.EndIndex - t.StartIndex + 1
	matchLength := 0
	for ; matchLength < length; matchLength++ {
		if data[t.StartIndex+matchLength] != data[currentIndex+matchLength] {
			break
		}
	}
	// matchLength == 0 is also handled here
	if matchLength < length {
		// split
		leftOccurrences := make([]int, len(t.Occurrences))
		copy(leftOccurrences, t.Occurrences)
		leftChildren := make([]*SuffixTree, len(t.Children))
		copy(leftChildren, t.Children)
		left := &SuffixTree{
			StartIndex:  t.StartIndex + matchLength,
			EndIndex:    t.EndIndex,
			Occurrences: leftOccurrences,
			Children:    leftChildren,
		}

		rightOccurrences := make([]int, 1)
		rightOccurrences[0] = startIndex
		right := &SuffixTree{
			StartIndex:  currentIndex + matchLength,
			EndIndex:    len(data) - 1,
			Occurrences: rightOccurrences,
			Children:    make([]*SuffixTree, 0),
		}

		t.Children = make([]*SuffixTree, 2)
		t.Children[0] = left
		t.Children[1] = right
		t.Occurrences = append(t.Occurrences, startIndex)
		t.EndIndex = t.StartIndex + matchLength - 1
	} else {
		if t.StartIndex <= t.EndIndex {
			t.Occurrences = append(t.Occurrences, startIndex)
		}

		if currentIndex+length >= len(data) {
			return
		}

		// go into children; if non matches, create one
		for _, child := range t.Children {
			if child.StartsWith(data) == data[currentIndex+length] {
				child.Insert(data, startIndex, currentIndex+length)
				return
			}
		}
		// no matching child found
		newChildOccurrences := make([]int, 1)
		newChildOccurrences[0] = startIndex
		newChild := &SuffixTree{
			StartIndex:  currentIndex + length,
			EndIndex:    len(data) - 1,
			Occurrences: newChildOccurrences,
			Children:    make([]*SuffixTree, 0),
		}
		t.Children = append(t.Children, newChild)
	}
}

// Creates a graph that can be used for occurrence optimization out of the
// suffix tree. The nodes of the graph are all characters in the input and
// for each node the outgoing edges (non-terminals starting there) are added.
func (t *SuffixTree) CreateGraph(graph Graph, nts *[]*NonTerminal, depth int) {
	startIndex := t.Occurrences[0]
	length := t.EndIndex - t.StartIndex + 1 + depth
	count := len(t.Occurrences)

	// check whether it is still a repeat
	if count <= 1 {
		return
	}

	newDepth := depth
	if t.StartIndex > t.EndIndex {
		length = 0
	} else {
		newDepth = length

		for j := t.EndIndex - t.StartIndex; j >= 0; j-- {
			if length-j < 2 {
				// repeats have to consits of at least 2 characters
				continue
			}

			// exclude non-terminals that never compress
			curLength := length - j
			curCount := len(t.Occurrences)
			mc := curCount*curLength - curCount - curLength - 1
			if mc <= 0 {
				continue
			}

			currentNt := &NonTerminal{
				Id:     len(*nts),
				Start:  startIndex,
				Length: curLength,
				Count:  curCount,
			}
			(*nts) = append(*nts, currentNt)

			// add the current non-terminal to all indexes where it occurs
			for _, i := range t.Occurrences {
				if graph[i] == nil {
					graph[i] = make([]CompNt, 0, 1)
				}
				graph[i] = append(graph[i], CompNt{
					Id:     currentNt.Id,
					Length: currentNt.Length,
				})
			}
		}
	}

	// recursivly visit all children
	for _, child := range t.Children {
		child.CreateGraph(graph, nts, newDepth)
	}
}
