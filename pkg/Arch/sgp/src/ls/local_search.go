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

package ls

import (
	"log"
	"oo"
	"sort"
	"sync/atomic"
)

const (
	Logging  = false
	addBound = -2 // bound for the heuristic benefit for adding non-terminals
)

var (
	d         []int
	p         []*oo.NonTerminal
	ntLengths []int
	ntCounts  []int

	alreadyChecked  []bool
	toCalculate     [][]int
	ntRemoveBenefit []int

	ntAddBenefit  []int
	inNt          []int
	addCandidates AddCandidateSlice

	next []int

	addScoreLimit = 1000 // very important for the performance

	OoCounter *int64 = new(int64)
)

type AddCandidate struct {
	Id      int
	Benefit int

	Ben int
	Len int
}

type AddCandidateSlice []*AddCandidate

func Init(file *oo.File, addCandidateNumber int) {
	d = make([]int, file.Size+1)
	p = make([]*oo.NonTerminal, file.Size+1)
	ntLengths = make([]int, file.NtCount)
	ntCounts = make([]int, file.NtCount)

	alreadyChecked = make([]bool, file.NtCount)
	toCalculate = make([][]int, file.NtCount)
	for i := range toCalculate {
		toCalculate[i] = make([]int, 0, 10)
	}
	ntRemoveBenefit = make([]int, file.NtCount)

	ntAddBenefit = make([]int, file.NtCount)
	inNt = make([]int, file.Size+1)
	addCandidates = make(AddCandidateSlice, 0, file.NtCount)

	next = make([]int, file.NtCount)

	addScoreLimit = addCandidateNumber
}

// Initial local search
// The main routine of the local search. Starts by going down and
// stops as soon as a downwards or upwards pass yields no
// improvement.
func LocalSearchDown(file *oo.File, currentChoice []bool, handleSolution func(*oo.File, *oo.Solution)) []bool {
	change := true

	for change {
		currentChoice, change = goDown(file, currentChoice, handleSolution)
		if !change {
			break
		}
		currentChoice, change = goUp(file, currentChoice, handleSolution)
	}

	return currentChoice
}

// The main routine of the local search. Starts by going upwards
// but doesn't stop if the first search upwards yields no improvement.
func LocalSearch(file *oo.File, currentChoice []bool) *oo.Individual {
	change := true
	first := true

	for change {
		currentChoice, change = goUp(file, currentChoice, nil)
		if !change && !first {
			break
		}
		first = false
		currentChoice, change = goDown(file, currentChoice, nil)
	}

	score := file.Score(currentChoice)
	return &oo.Individual{score, currentChoice}
}

// Going down: removes non-terminals as long as there is a change
func goDown(file *oo.File, currentChoice []bool, handleSolution func(*oo.File, *oo.Solution)) ([]bool, bool) {
	atLeastOneChange := false

	change := true
	for change {
		currentChoice, change = removeNtSimple(file, currentChoice)
		if change {
			atLeastOneChange = true
			if handleSolution != nil {
				score := file.Score(currentChoice)
				handleSolution(file, &oo.Solution{
					Score:   score,
					Choice:  currentChoice,
					NoScore: false,
				})
			}
			if Logging {
				score := file.Score(currentChoice)
				log.Printf("Ls down: %v\n", score)
			}
		}
	}

	change = true
	for change {
		currentChoice, change = removeNtPerfect(file, currentChoice)
		if change {
			atLeastOneChange = true
			if handleSolution != nil {
				score := file.Score(currentChoice)
				handleSolution(file, &oo.Solution{
					Score:   score,
					Choice:  currentChoice,
					NoScore: false,
				})
			}
			if Logging {
				score := file.Score(currentChoice)
				log.Printf("Ls down: %v\n", score)
			}
		}
	}

	return currentChoice, atLeastOneChange
}

// Going up: adds non-terminals as long as there is a change
func goUp(file *oo.File, currentChoice []bool, handleSolution func(*oo.File, *oo.Solution)) ([]bool, bool) {
	atLeastOneChange := false

	change := true
	for change {
		currentChoice, change = addNtByRanking(file, currentChoice)
		if change {
			atLeastOneChange = true
			if handleSolution != nil {
				score := file.Score(currentChoice)
				handleSolution(file, &oo.Solution{
					Score:   score,
					Choice:  currentChoice,
					NoScore: false,
				})
			}
			if Logging {
				score := file.Score(currentChoice)
				log.Printf("Ls up:   %v\n", score)
			}
		}
	}

	return currentChoice, atLeastOneChange
}

// Removes non-terminal that don't compress, i.e. their compression rate is smaller or equal to zero
func removeNtSimple(file *oo.File, choice []bool) ([]bool, bool) {
	for i := range ntCounts {
		ntCounts[i] = 0
	}
	for i := range ntLengths {
		ntLengths[i] = 0
	}

	file.ShortestPathCount(d, p, choice, -1, ntCounts)
	for i, c := range choice {
		if c {
			ntLengths[i] = file.ShortestPathCount(d, p, choice, i, ntCounts)
		}
	}
	atomic.AddInt64(OoCounter, 1)

	// non-terminals with a compression of 0 or less are removed
	change := false
	for i, c := range choice {
		if c {
			ntC := ntCounts[i]*ntLengths[i] - ntCounts[i] - ntLengths[i] - 1
			if ntC <= 0 {
				choice[i] = false
				change = true
			}
		}
	}
	return choice, change
}

// Computes the score function for every used non-terminal in a clever way
func removeNtPerfect(file *oo.File, choice []bool) ([]bool, bool) {
	for i := range ntLengths {
		ntLengths[i] = 0
	}

	lengthOfMainNt := file.GetOptimalLength(d, choice, -1)
	activeNtCount := 0
	for i, c := range choice {
		if c {
			ntLengths[i] = file.GetOptimalLength(d, choice, i)
			activeNtCount++
		}
	}
	atomic.AddInt64(OoCounter, 1)

	for i := range alreadyChecked {
		alreadyChecked[i] = false
	}
	for i := range toCalculate {
		toCalculate[i] = toCalculate[i][:0]
	}

	file.ShortestPathOccurences(d, p, alreadyChecked, choice, -1, ntLengths, toCalculate, activeNtCount)
	for i, c := range choice {
		if c {
			file.ShortestPathOccurences(d, p, alreadyChecked, choice, i, ntLengths, toCalculate, activeNtCount)
		}
	}
	atomic.AddInt64(OoCounter, 1)

	for i := range ntRemoveBenefit {
		ntRemoveBenefit[i] = 0
	}
	removeNtBenefit(file, choice, ntLengths, toCalculate, ntRemoveBenefit, lengthOfMainNt)

	// non-terminals with a compression of 0 or less are removed
	change := false
	for i, c := range choice {
		if c {
			if ntRemoveBenefit[i] >= 0 {
				change = true
				choice[i] = false
			}
		}
	}
	return choice, change
}

// Adds non-terminal by first computing an heuristic benefit and then evaluating the score functions
// in the order given by the heuristic benefits
func addNtByRanking(file *oo.File, choice []bool) ([]bool, bool) {
	score := file.Score(choice)

	// slice of integers that can be accessed concurrently
	for i := range ntAddBenefit {
		ntAddBenefit[i] = 0
	}

	file.ShortestPathAdd(d, p, inNt, choice, -1, ntAddBenefit, next)
	for i, c := range choice {
		if c {
			file.ShortestPathAdd(d, p, inNt, choice, i, ntAddBenefit, next)
		}
	}
	atomic.AddInt64(OoCounter, 1)

	// create a list of benefits that above the given bound
	addCandidates := addCandidates[:0]
	for i, c := range choice {
		if !c {
			if ntAddBenefit[i] > addBound+2 { // length is at least 2
				length := file.GetOptimalLength(d, choice, i)
				// ntAddBenefit[i] is only a heuristic value
				curBenefit := ntAddBenefit[i] - length - 1
				if curBenefit >= addBound {
					addCandidates = append(addCandidates, &AddCandidate{
						Id:      i,
						Benefit: curBenefit,
						Ben:     ntAddBenefit[i],
						Len:     length,
					})
				}
			}
		}
	}
	atomic.AddInt64(OoCounter, 1)

	sort.Sort(addCandidates)

	// check the score functions for all candidates
	for i, ac := range addCandidates {
		j := ac.Id

		choice[j] = true
		curBenefit := score - file.Score(choice)
		if curBenefit > 0 {
			if Logging {
				log.Printf("Ls add   %v, %v\n", i, curBenefit)
			}
			return choice, true /* change */
		}
		choice[j] = false

		if i > addScoreLimit {
			break
		}
	}

	return choice, false /* change */
}

// Calculation of the benefit of removing a non-terminal (in the given range)
func removeNtBenefit(file *oo.File, choice []bool, ntLength []int, toCalculate [][]int, ntRemoveBenefit []int, lengthOfMainNt int) {
	for j, c := range choice {
		if c {
			if toCalculate[j] == nil {
				ntRemoveBenefit[j] = ntLength[j] + 1
				continue
			}

			choice[j] = false
			for _, k := range toCalculate[j] {
				if k == -1 {
					curLength := file.GetOptimalLength(d, choice, -1)
					ntRemoveBenefit[j] += lengthOfMainNt - curLength
				} else {
					curLength := file.GetOptimalLength(d, choice, k)
					ntRemoveBenefit[j] += ntLength[k] - curLength
				}
			}
			choice[j] = true
			ntRemoveBenefit[j] += ntLength[j] + 1
		}
	}
	atomic.AddInt64(OoCounter, 1)
}

// Len, Less and Swap are needed so that the slice can be sorted
func (a AddCandidateSlice) Len() int {
	return len(a)
}

func (a AddCandidateSlice) Less(i, j int) bool {
	return a[i].Benefit > a[j].Benefit
}

func (a AddCandidateSlice) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}
