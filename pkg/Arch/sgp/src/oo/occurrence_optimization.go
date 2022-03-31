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

import (
	"sync/atomic"
)

var (
	OoCounter *int64 = new(int64)
)

// Here is the actual computation of the shortest path distances.
// p can be nil
func (file *File) GetDistances(d []int, p []*NonTerminal, choice []bool, curNtId int) (start, end int) {
	start = 0
	end = file.Size
	if curNtId >= 0 {
		start = file.NonTerminals[curNtId].Start
		end = start + file.NonTerminals[curNtId].Length
	}

	for i := start; i < end+1; i++ {
		// init with a distance that can always be beaten (same effect as infinifty)
		d[i] = end + 1
	}
	d[start] = 0

	for i := start; i < end; i++ {
		// check whether a simple step forward leads to a shorter path
		dInc := d[i] + 1
		if dInc < d[i+1] {
			d[i+1] = dInc
			if p != nil {
				p[i+1] = nil
			}
		}
		for _, nt := range file.Graph[i] {
			if !choice[nt.Id] || nt.Id == curNtId || i+nt.Length > end {
				continue
			}

			ntEnd := i + nt.Length

			// check whether the current non-terminals leads to a shorter path
			if dInc < d[ntEnd] {
				d[ntEnd] = dInc
				if p != nil {
					p[ntEnd] = file.NonTerminals[nt.Id]
				}
			}
		}
	}
	return
}

// Reconstruction of the shortest path with the given shortest distances. Calls the given
// functions on each step of the shortest path so that this function can be used as a
// generic helper.
func (file *File) PathReconstruction(p []*NonTerminal, end, length int, fData func(int, int, uint32),
	fNt func(int, int, *NonTerminal)) {
	pos := end
	for i := length - 1; i >= 0; i-- {
		if p[pos] == nil {
			pos--
			fData(i, pos, file.Data[pos])
		} else {
			currentNt := p[pos]
			pos -= currentNt.Length
			fNt(i, pos, currentNt)
		}
	}
}

// Returns the shortest path in a non-terminal (in the internal encoding).
func (file *File) ShortestPath(activeNts []bool, ntNum int, ntNumbering []int) []uint32 {
	d := make([]int, file.Size+1)
	p := make([]*NonTerminal, file.Size+1)

	_, end := file.GetDistances(d, p, activeNts, ntNum)

	result := make([]uint32, d[end])
	fData := func(i, pos int, data uint32) {
		result[i] = data
	}
	fNt := func(i, pos int, nt *NonTerminal) {
		result[i] = 257 + uint32(ntNumbering[nt.Id])
	}
	file.PathReconstruction(p, end, d[end], fData, fNt)
	return result
}

// Returns only the length of the shortest path in a non-terminal.
func (file *File) GetOptimalLength(d []int, choice []bool, ntNum int) int {
	_, end := file.GetDistances(d, nil, choice, ntNum)
	return d[end]
}

// Returns the shortest path in a non-terminal (in the internal encoding) and additionally computes
// how often each non-terminal is used in this shortest path.
func (file *File) ShortestPathCount(d []int, p []*NonTerminal, activeNts []bool, ntNum int, ntCounts []int) int {
	_, end := file.GetDistances(d, p, activeNts, ntNum)

	fData := func(i, pos int, data uint32) {}
	fNt := func(i, pos int, nt *NonTerminal) {
		ntCounts[nt.Id]++
	}
	file.PathReconstruction(p, end, d[end], fData, fNt)
	return d[end]
}

// Determines which other non-terminals are used in the shortest path of the given non-terminal.
// The combined results can then be used to minimize the computations for determining the score
// without a non-terminal that was used before.
func (file *File) ShortestPathOccurences(d []int, p []*NonTerminal, alreadyChecked []bool, activeNts []bool, ntNum int,
	ntLength []int, toCalculate [][]int, activeNtCount int) {
	for i := 0; i < len(alreadyChecked); i++ {
		alreadyChecked[i] = false
	}

	_, end := file.GetDistances(d, p, activeNts, ntNum)

	fData := func(i, pos int, data uint32) {}
	fNt := func(i, pos int, nt *NonTerminal) {
		if alreadyChecked[nt.Id] {
			return
		}
		toCalculate[nt.Id] = append(toCalculate[nt.Id], ntNum)
		alreadyChecked[nt.Id] = true
	}
	file.PathReconstruction(p, end, d[end], fData, fNt)
}

// Computes the heuristic benefits of adding a non-terminal for all non-terminals that are not used.
// It is basically computed whether adding at any possible point would shorten the shortest path.
// However, this is only done if the 'new non-terminal' does not interfere with already used ones.
func (file *File) ShortestPathAdd(d []int, p []*NonTerminal, inNt []int, choice []bool, ntNum int, ntAddBenefit []int, next []int) {
	start, end := file.GetDistances(d, p, choice, ntNum)

	for i := start; i <= end; i++ {
		inNt[i] = -1
	}

	// Consider NTs that overlap themselves only once at
	// overlapping places
	for i := range next {
		next[i] = -1
	}

	fData := func(i, pos int, data uint32) {}
	fNt := func(i, pos int, nt *NonTerminal) {
		inNt[pos] = nt.Id
		for j := 1; j < nt.Length; j++ {
			inNt[pos+j] = -2
		}
		if inNt[pos+nt.Length] == -1 {
			inNt[pos+nt.Length] = -3
		}
	}
	file.PathReconstruction(p, end, d[end], fData, fNt)

	for i := start; i < end; i++ {
		for _, nt := range file.Graph[i] {
			if choice[nt.Id] || nt.Id == ntNum || i+nt.Length > end {
				continue
			}

			ntEnd := i + nt.Length

			curBenefit := d[ntEnd] - d[i] - 1
			if curBenefit <= 0 {
				continue
			}

			// the condition for the benefit heuristic: start and end are not covered by used non-terminals
			// and adding at that point would shorten the shortest path
			if inNt[i] != -2 && inNt[ntEnd] != -2 {
				if i < next[nt.Id] {
					continue
				}

				ntAddBenefit[nt.Id] += curBenefit
				next[nt.Id] = ntEnd
			}
		}
	}
}

// Additional to the score, all shortest paths are computed.
func (file *File) ScoreExtended(choice []bool, ntIds []int) ([]uint32, int) {
	atomic.AddInt64(OoCounter, 1)
	result := make([]uint32, 0, file.Size)

	// create a numberin table so that the used non-terminals are
	// numbered 0 to (number of non-terminals used) - 1
	ntNumbering := make([]int, file.NtCount)
	number := 1
	for i, c := range choice {
		if c {
			if !IsValidNt(number) {
				number++
			}
			ntNumbering[i] = number
			ntIds[number] = i
			number++
		} else {
			ntNumbering[i] = -1
		}
	}

	curNtData := file.ShortestPath(choice, -1, ntNumbering)
	copy(result, curNtData)
	result = append(result, curNtData...)

	for j, c := range choice {
		if c {
			curNtData = file.ShortestPath(choice, j, ntNumbering)
			result = append(result, 256) // seperator
			result = append(result, curNtData...)
		}
	}

	// returns complete encoded result and the score
	return result, len(result) + 1
}

func (file *File) PlainReconstruction(choice []bool) []uint32 {
	result := make([]uint32, 0, file.Size)

	// create a numberin table so that the used non-terminals are
	// numbered 0 to (number of non-terminals used) - 1
	ntNumbering := make([]int, file.NtCount)
	number := 1
	for i, c := range choice {
		if c {
			ntNumbering[i] = number
			number++
		} else {
			ntNumbering[i] = -1
		}
	}

	curNtData := file.ShortestPath(choice, -1, ntNumbering)
	copy(result, curNtData)
	result = append(result, curNtData...)

	for j, c := range choice {
		if c {
			curNtData = file.ShortestPath(choice, j, ntNumbering)
			result = append(result, 256) // seperator
			result = append(result, curNtData...)
		}
	}

	// returns complete encoded result
	return result
}

func (file *File) ScoreWithReconstruction(currentChoice []bool) ([]byte, int) {
	ntIds := make([]int, file.NtCount)
	result, _ := file.ScoreExtended(currentChoice, ntIds)

	// returns complete encoded result and the score
	return EncodeGrammar(result, file.Size)
}

// Non-concurrent score function
func (file *File) Score(choice []bool) int {
	atomic.AddInt64(OoCounter, 1)
	d := make([]int, file.Size+1)
	score := 1 + file.GetOptimalLength(d, choice, -1)
	for j, c := range choice {
		if c {
			score += 1 + file.GetOptimalLength(d, choice, j)
		}
	}
	return score
}

// Returns the path used by ACO, only for the initial non-terminal for now
func (file *File) Path(choice []bool) []int {
	d := make([]int, file.Size+1)
	p := make([]*NonTerminal, file.Size+1)
	_, end := file.GetDistances(d, p, choice, -1)
	path := make([]int, file.Size)
	for i := range path {
		path[i] = -1
	}
	fData := func(i, pos int, data uint32) {
		path[pos] = len(file.Graph[pos])
	}
	fNt := func(i, pos int, nt *NonTerminal) {
		for j, n := range file.Graph[pos] {
			if nt.Id == n.Id {
				path[pos] = j
				return
			}
		}
		panic("no matching non-terminal at the given position")
	}
	file.PathReconstruction(p, end, d[end], fData, fNt)
	return path
}

// Completes the choice by computing the MGP on the chosen non-terminals and adds the used non-terminals
func (file *File) CompleteChoice(choice []bool) []bool {
	newChoice := make([]bool, len(choice))
	d := make([]int, file.Size+1)
	p := make([]*NonTerminal, file.Size+1)
	fData := func(i, pos int, data uint32) {}
	fNt := func(i, pos int, nt *NonTerminal) {
		newChoice[nt.Id] = true
	}
	for i, c := range choice {
		if c {
			_, end := file.GetDistances(d, nil, choice, i)
			file.PathReconstruction(p, end, d[end] /* length */, fData, fNt)
			newChoice[i] = true
		}
	}
	return newChoice
}

func (file *File) ScoreCleanUp(choice []bool) (int, []bool) {
	d := make([]int, file.Size+1)
	p := make([]*NonTerminal, file.Size+1)
	ntCounts := make([]int, len(choice))
	ntLengths := make([]int, len(choice))

	score := 1 + file.ShortestPathCount(d, p, choice, -1, ntCounts)
	for i, c := range choice {
		if c {
			ntLengths[i] = file.ShortestPathCount(d, p, choice, i, ntCounts)
			score += 1 + ntLengths[i]
		}
	}

	// non-terminals with a compression of 0 or less are removed
	for i, c := range choice {
		if c {
			ntC := ntCounts[i]*ntLengths[i] - ntCounts[i] - ntLengths[i] - 1
			if ntC <= 0 {
				choice[i] = false
			}
		}
	}

	return score, choice
}

// Transformation from the internal encoding to the specified one
func EncodeGrammar(data []uint32, capacity int) ([]byte, int) {
	output := make([]byte, 0, capacity)
	// always starts with the first non-terminal (#00)
	output = append(output, byte(35)) // #
	output = append(output, byte(0))
	output = append(output, byte(0))

	score := 1
	ntCounter := uint32(1)
	for _, d := range data {
		if d < 256 {
			// check for \, N, #
			if d == 92 || d == 78 || d == 35 {
				output = append(output, byte(92))
			}
			output = append(output, byte(d))
			score++
		} else if d == 256 {
			if !IsValidNt(int(ntCounter)) {
				ntCounter++
			}
			high, low := encodeNonTerminal(ntCounter)
			output = append(output, byte(35)) // #
			output = append(output, high)
			output = append(output, low)
			ntCounter++
			score++
		} else {
			ntNum := d - 257
			high, low := encodeNonTerminal(ntNum)
			output = append(output, byte(78)) // N
			output = append(output, high)
			output = append(output, low)
			score++
		}
	}
	return output, score
}

// Avoid critical NT encodings
func IsValidNt(ntCounter int) bool {
	if ntCounter == 35 || ntCounter == 78 || ntCounter == 92 || ntCounter == 10 || ntCounter == 13 {
		return false
	}
	ntCounter %= 256
	if ntCounter == 35 || ntCounter == 78 || ntCounter == 92 || ntCounter == 10 || ntCounter == 13 {
		return false
	}
	return true
}

// Encodes a non-terminal with two bytes
func encodeNonTerminal(nt uint32) (byte, byte) {
	high := byte(nt / 256)
	low := byte(nt % 256)
	return high, low
}

func (file *File) GetDistancesPart(d []int, choice []bool, curNtId, start, end int) {
	for i := start; i < end; i++ {
		// check whether a simple step forward leads to a shorter path
		dInc := d[i] + 1
		if dInc < d[i+1] {
			d[i+1] = dInc
		}
		for _, nt := range file.Graph[i] {
			if !choice[nt.Id] || nt.Id == curNtId || i+nt.Length > end {
				continue
			}

			ntEnd := i + nt.Length

			// check whether the current non-terminals leads to a shorter path
			if dInc < d[ntEnd] {
				d[ntEnd] = dInc
			}
		}
	}
}
