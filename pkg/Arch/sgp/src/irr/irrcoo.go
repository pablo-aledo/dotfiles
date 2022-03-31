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
	"oo"
	"qsufsort"
)

// Compute several IRRCOO runs in parallel and return the best result
func AllIrrCoo(file *oo.File, maxModChoice int) (int, []bool) {
	var bestResult *IrrResult = nil
	for i := 0; i < maxModChoice; i++ {
		curResult := IrrCoo(file, MC, i)
		log.Printf("IRRCOO, run %d: %d\n", i+1, curResult.Score)
		if bestResult == nil || curResult.Score < bestResult.Score {
			bestResult = curResult
		}
	}
	return bestResult.Score, bestResult.Choice
}

// The main function of the IRRCOO algorithm
// Version without the cleanup
func IrrCoo(file *oo.File, scoreFn func(int, int) int, choiceMod int) *IrrResult {
	data := make([]uint32, file.Size)
	copy(data, file.Data)
	score := -1

	// one non-terminal always exists
	ntCounter := 1
	irrNts := make([]int, file.NtCount)
	currentChoice := make([]bool, file.NtCount)

	sa := make([]int, len(data))
	inv := make([]int, len(data))
	ss := qsufsort.NewSuffixSortable(sa, inv, 1)
	ignore := make([]bool, len(sa))

	for i := 0; i < 65536-5; /* max number of non-terminals */ i++ {
		replace := getNextNonTerminal(ss, nil, ignore, data, ntCounter, scoreFn, choiceMod)
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

		// search the matching non-terminal and set the corresponding bit
		index := searchNt(file, irrNts, oldPart, ntCounter)
		currentChoice[index] = true

		//_, currentChoice = file.ScoreCleanUp(currentChoice) no change for grammar.lsp
		data, score = file.ScoreExtended(currentChoice, irrNts)
	}

	return &IrrResult{score, currentChoice}
}
