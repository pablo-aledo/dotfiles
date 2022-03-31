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

package irr

import (
	"log"
	"math/rand"
	"oo"
	"qsufsort"
)

type IrrResult struct {
	Score  int
	Choice []bool
}

// Only MC is used in this program; ML and MO are listed for completeness
func MC(length, occurrences int) int {
	return length*occurrences - length - occurrences - 1
}

func ML(length, occurrences int) int {
	return length
}

func MO(length, occurrences int) int {
	return occurrences
}

// Best result of several IRR + OO runs
func AllIrr(file *oo.File, maxModChoice int) (int, []bool) {
	bestResult := &IrrResult{
		Score:  file.Size + 1,
		Choice: make([]bool, file.NtCount),
	}
	for i := 0; i < maxModChoice; i++ {
		curResult := Irr(file, nil, MC, i)
		log.Printf("IRR + OO, run %d: %d\n", i+1, curResult.Score)
		if curResult.Score < bestResult.Score {
			bestResult = curResult
		}
	}
	return bestResult.Score, bestResult.Choice
}

func RandomizedIrr(file *oo.File, rnd *rand.Rand, tries int) (int, []bool) {
	bestResult := &IrrResult{
		Score:  file.Size + 1,
		Choice: make([]bool, file.NtCount),
	}
	for i := 0; i < tries; i++ {
		curResult := Irr(file, rnd, MC, -1)
		log.Printf("IRR + OO, run %d: %d\n", i+1, curResult.Score)
		if curResult.Score < bestResult.Score {
			bestResult = curResult
		}
	}
	return bestResult.Score, bestResult.Choice
}

// The main function of the IRR algorithm
func Irr(file *oo.File, rnd *rand.Rand, scoreFn func(int, int) int, choiceMod int) *IrrResult {
	data := make([]uint32, file.Size)
	copy(data, file.Data)

	// one non-terminal always exists
	ntCounter := 1
	irrNts := make([]int, file.NtCount)
	currentChoice := make([]bool, file.NtCount)

	sa := make([]int, len(data))
	inv := make([]int, len(data))
	ss := qsufsort.NewSuffixSortable(sa, inv, 1)
	ignore := make([]bool, len(sa))

	for i := 0; i < 65536-5; /* max number of non-terminals */ i++ {
		replace := getNextNonTerminal(ss, rnd, ignore, data, ntCounter, scoreFn, choiceMod)
		if replace == nil {
			// nothing more to replace
			break
		}

		newNonTerminal := make([]uint32, 1)
		if !oo.IsValidNt(ntCounter) {
			ntCounter++
		}
		newNonTerminal[0] = 257 + uint32(ntCounter)
		ntCounter++

		oldPart := make([]uint32, len(replace))
		copy(oldPart, replace)
		data = replaceWithNonTerminal(data, oldPart, newNonTerminal)

		// search the matching non-terminal and set the corresponding bit
		index := searchNt(file, irrNts, oldPart, ntCounter)
		currentChoice[index] = true

		data = append(data, 256) // seperator
		data = append(data, oldPart...)
	}
	//log.Printf("data length %d (%d)", len(data), ntCounter)
	ooScore := file.Score(currentChoice)

	return &IrrResult{ooScore, currentChoice}
}

// Very simple replace, but it is in place
func replaceWithNonTerminal(old, oldPart, newPart []uint32) []uint32 {
	new := old // in place
	c := oldPart[0]
	n := len(oldPart)
	newI := 0
	for i := 0; i < len(old); i, newI = i+1, newI+1 {
		new[newI] = old[i]

		// search
		if old[i] == c && i+n <= len(old) {
			match := true
			for j := 0; j < n && match; j++ {
				if old[i+j] != oldPart[j] {
					match = false
				}
			}

			if match {
				// replace
				for k := 0; k < len(newPart); k, newI = k+1, newI+1 {
					new[newI] = newPart[k]
				}
				newI--
				i += n - 1
			}
		}
	}
	return new[0:newI]
}

// Reconstruction of non-terminal from the already created non-terminals
func reconstructNt(file *oo.File, irrNts []int, nt []uint32) []uint32 {
	res := make([]uint32, 0, 100)

	for _, n := range nt {
		if n < 256 {
			res = append(res, n)
		} else {
			curNt := file.NonTerminals[irrNts[int(n)-257]]
			res = append(res, file.Data[curNt.Start:curNt.Start+curNt.Length]...)
		}
	}
	return res
}

// Compares a given array with a part of the original data
// The part of the original data is a non-terminal and the
// given array is the result of one IRR-MC step.
func compare(file *oo.File, start, length int, out []uint32) int {
	k := 0
	match := true
	for ; k < len(out) && k < length; k++ {
		if file.Data[start+k] != out[k] {
			match = false
			break
		}
	}
	if match {
		if length == len(out) {
			return 0 // equal
		} else if len(out) < length {
			return -1 // less
		} else {
			return 1 // greater
		}
	}
	if out[k] < file.Data[start+k] {
		return -1 // less
	} else {
		return 1 // greater
	}
	return 0
}

// Returns the index of the non-terminals that corresponds to the given repeat
func searchNt(file *oo.File, irrNts []int, oldPart []uint32, ntCounter int) int {
	reconstructedNt := reconstructNt(file, irrNts, oldPart)

	// binary search for the matching non-terminal
	for l, r := 0, file.NtCount-1; l <= r; {
		m := l + (r-l+1)/2
		nt := file.NonTerminals[m]
		res := compare(file, nt.Start, nt.Length, reconstructedNt)
		if res == 0 {
			irrNts[ntCounter-1] = m
			return m
		} else if res < 0 {
			r = m - 1
		} else {
			l = m + 1
		}
	}
	panic("no matching non-terminal was found")
}

// Returns the repeat with the highest score according to the given score function
func getNextNonTerminal(ss *qsufsort.SuffixSortable, rnd *rand.Rand, ignore []bool, data []uint32, ntCount int, scoreFn func(int, int) int, choiceMod int) []uint32 {
	ss.Resize(len(data))
	sa := qsufsort.Qsufsort(ss, data, ntCount)

	maxScore := 0
	indexOfMax := -1
	lengthOfMax := -1

	for i := 0; i < len(sa); i++ {
		ignore[i] = false
	}

	repeats := 0
	// only used by the randomized variant
	maxRepeats := make([]int, 0, 20)
	maxLengths := make([]int, 0, 20)
	for i := 0; i < len(sa)-1; i++ {
		if ignore[i] {
			continue
		}

		j := sa[i]
		// non-overlapping criterium: (j < sa[i+1] || k < sa[i])
		for k := sa[i+1]; j < len(data) && k < len(data) && (j < sa[i+1] || k < sa[i]); j, k = j+1, k+1 {
			if data[j] != data[k] || data[j] == 256 {
				break
			}
		}

		length := j - sa[i]
		if length < 2 {
			continue
		}
		count := 2
		repeats++

		match := true
		k := i + 2
		// search downwards
		for ; match && k < len(sa); k++ {
			if sa[k]+length >= len(data) {
				match = false
			} else {
				for l := length - 1; l >= 0 && match; l-- {
					if data[sa[k]+l] != data[sa[i]+l] {
						match = false
					}
				}
			}
			if match {
				if data[sa[k-1]+length] != data[sa[k]+length] {
					ignore[k-1] = true
				}
				count++
			}
		}

		// search upwards
		match = true
		for h := i - 1; match && h >= 0; h-- {
			if sa[h]+length >= len(data) {
				match = false
			}
			if sa[h]+length >= len(data) {
				match = false
			} else {
				for l := length - 1; match && l >= 0; l-- {
					if data[sa[h]+l] != data[sa[i]+l] {
						match = false
					}
				}
			}
			if match {
				count++
			}
		}

		score := scoreFn(length, count)

		if choiceMod >= 0 {
			// sometimes it is better to not choose the first maximum
			if score > maxScore || (choiceMod > 0 && score >= maxScore && repeats%choiceMod == 0) {
				maxScore = score
				indexOfMax = sa[i]
				lengthOfMax = length
			}
		} else {
			if score > maxScore {
				maxScore = score
				maxRepeats = maxRepeats[:0]
				maxLengths = maxLengths[:0]
				maxRepeats = append(maxRepeats, sa[i])
				maxLengths = append(maxLengths, length)
			} else if score == maxScore {
				maxRepeats = append(maxRepeats, sa[i])
				maxLengths = append(maxLengths, length)
			}
		}
	}

	if maxScore <= 0 {
		return nil
	} else if choiceMod < 0 {
		i := rnd.Intn(len(maxRepeats))
		indexOfMax = maxRepeats[i]
		lengthOfMax = maxLengths[i]
	}
	return data[indexOfMax : indexOfMax+lengthOfMax]
}
