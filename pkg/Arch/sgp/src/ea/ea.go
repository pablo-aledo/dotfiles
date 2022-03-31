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

package ea

import (
	"container/list"
	"fmt"
	"hash/adler32"
	"log"
	"ls"
	"math/rand"
	"oo"
	"strconv"
	"time"
)

const (
	generationSize                    = 30
	childCount                        = 24
	crossoverProbability              = 0.7
	mutationAfterCrossoverProbability = 0.8
	lsPerGeneration                   = 1   // number of local searches per generation
	localSearchBound                  = 1.2 // maximum difference to the best-so-far solution to get selected for a local search
	mutationDivisor                   = 2   // the expected number of bit flips in a single mutation
	initialMutationDivisor            = 2   // the expected number of bit flips in the initial mutation
)

// A classical EA combined with local searches running concurrently
func Run(file *oo.File, initialSolution *oo.Individual, rnd *rand.Rand, handleSolution func(*oo.File, *oo.Solution), startTime time.Time, localSearchActive, csv bool, number int) {
	// init first generation
	generation := make([]*oo.Individual, generationSize)
	generation[0] = initialSolution
	for i := 1; i < generationSize; i++ {
		curChoice := make([]bool, file.NtCount)
		copy(curChoice, initialSolution.Choice)
		// initial mutationprob := ntCount/mutationDivisor
		prob := file.NtCount / initialMutationDivisor
		if prob <= 1 {
			prob = 2
		}
		for b := 0; b < file.NtCount; b++ {
			if rnd.Intn(prob) == 0 {
				curChoice[b] = !curChoice[b]
			}
		}
		curScore := file.Score(curChoice)
		generation[i] = &oo.Individual{curScore, curChoice}
	}

	maxScore := 0
	memory := make(map[uint32]bool)

	globalBestScore := initialSolution.Score

	// evolution
	for i := 1; ; i++ {
		scoreSum := 0
		for j := 0; j < generationSize; j++ {
			if generation[j].Score > maxScore {
				maxScore = generation[j].Score
			}
		}
		for j := 0; j < generationSize; j++ {
			scoreSum += maxScore - generation[j].Score
		}

		newGeneration := make([]*oo.Individual, generationSize)
		pos := 0
		// generate children
		for j := 0; j < childCount/2; j++ {
			Reproduce(rnd, pos, maxScore, generation, newGeneration, scoreSum, file)
			pos += 2
		}

		// fill the rest with the fittest
		bestScore := 0
		alreadyUsedBest := list.New()
		for j := pos; j < generationSize; j++ {
			best := generation[0]
			for k := 0; k < generationSize; k++ {
				if generation[k].Score < best.Score && !listContains(alreadyUsedBest, generation[k]) {
					best = generation[k]
				}
			}
			newGeneration[j] = best
			alreadyUsedBest.PushBack(best)

			if alreadyUsedBest.Len() == 1 {
				bestScore = best.Score
				if bestScore < globalBestScore {
					globalBestScore = bestScore
					handleSolution(file, &oo.Solution{
						Score:   best.Score,
						Choice:  best.Choice,
						NoScore: false,
					})

					if csv {
						minutes := int(time.Now().Sub(startTime).Minutes())
						// number, iteration, #OO, minutes, current best
						fmt.Printf("%d, %d, %d, %d, %d\n", number, i, *oo.OoCounter+*ls.OoCounter, minutes, globalBestScore)
					}
				}

				if !csv {
					log.Printf("Generation %d, best: %d    %s  #OO: %d   (memory: %v)\n", i,
						best.Score, getTime(startTime), *oo.OoCounter+*ls.OoCounter, len(memory))
				}
			}
		}

		// start a local search
		for j := 0; localSearchActive && j < lsPerGeneration; j++ {
			pos := rnd.Intn(generationSize)
			localSearchInit := newGeneration[pos]

			// find an individual in the current generation that has a score below the limit
			// and no local search has been applied before on it
			emergencyBreak := 0
			for ; (float64(localSearchInit.Score) > float64(bestScore)*localSearchBound || memory[getHash(localSearchInit.Choice)]) && emergencyBreak < generationSize; emergencyBreak++ {
				pos = rnd.Intn(generationSize)
				localSearchInit = newGeneration[pos]
			}
			if emergencyBreak >= generationSize {
				break
			}

			// remeber the choice so that a local search is not applied again on it
			memory[getHash(localSearchInit.Choice)] = true
			lsResult := ls.LocalSearch(file, localSearchInit.Choice)
			if lsResult.Score <= generation[pos].Score {
				newGeneration[pos] = lsResult
			}
		}

		generation = newGeneration
	}
}

// Generates two new children by using crossover and mutation
func Reproduce(rnd *rand.Rand, pos int, maxScore int, generation []*oo.Individual, newGeneration []*oo.Individual,
	scoreSum int, file *oo.File) {

	parent1 := GetParent(rnd, maxScore, generation, scoreSum)
	parent2 := GetParent(rnd, maxScore, generation, scoreSum)

	if rnd.NormFloat64() < crossoverProbability {
		// crossover
		parent1, parent2 = GetCrossover(rnd, file, parent1, parent2)

		if rnd.NormFloat64() < mutationAfterCrossoverProbability {
			// mutation
			child1 := GetMutation(rnd, file, parent1)
			child2 := GetMutation(rnd, file, parent2)
			newGeneration[pos] = child1
			pos++
			newGeneration[pos] = child2
			pos++
		} else {
			// no change
			newGeneration[pos] = parent1
			pos++
			newGeneration[pos] = parent2
			pos++
		}
	} else {
		// only mutation
		child1 := GetMutation(rnd, file, parent1)
		child2 := GetMutation(rnd, file, parent2)
		newGeneration[pos] = child1
		pos++
		newGeneration[pos] = child2
		pos++
	}
}

// Putting together the used non-terminals but not throwing any out for the first child
// and doing the opposite for the second child
func GetCrossover(rnd *rand.Rand, file *oo.File, parent1, parent2 *oo.Individual) (*oo.Individual, *oo.Individual) {
	ntCount := len(parent1.Choice)

	childChoice1 := make([]bool, ntCount)
	childChoice2 := make([]bool, ntCount)
	for i := 0; i < ntCount; i++ {
		if parent1.Choice[i] && parent2.Choice[i] {
			childChoice1[i] = true
		}
		if parent1.Choice[i] || parent2.Choice[i] {
			childChoice2[i] = true
		}
	}
	childScore1 := file.Score(childChoice1)
	childScore2 := file.Score(childChoice2)

	return &oo.Individual{childScore1, childChoice1}, &oo.Individual{childScore2, childChoice2}
}

// Mutation by flipping bits with a probability depending on the input size
func GetMutation(rnd *rand.Rand, file *oo.File, parent *oo.Individual) *oo.Individual {
	ntCount := len(parent.Choice)
	childChoice := make([]bool, ntCount)
	copy(childChoice, parent.Choice)
	prob := ntCount / mutationDivisor
	if prob <= 1 {
		prob = 2
	}
	for b := 0; b < ntCount; b++ {
		if rnd.Intn(prob) == 0 {
			childChoice[b] = !childChoice[b]
		}
	}
	childScore := file.Score(childChoice)
	return &oo.Individual{childScore, childChoice}
}

// Simple roulette wheel selection
func GetParent(rnd *rand.Rand, maxScore int, generation []*oo.Individual, scoreSum int) *oo.Individual {
	r := rnd.Intn(scoreSum)
	sum := 0
	for i := 0; i < len(generation); i++ {
		if r >= sum && r < sum+(maxScore-generation[i].Score) {
			return generation[i]
		}
		sum += maxScore - generation[i].Score
	}
	return generation[len(generation)-1]
}

// Computes the Adler32 hash of the given choice
func getHash(choice []bool) uint32 {
	b := make([]byte, len(choice)/8+1)
	for i := 0; i < len(choice); i++ {
		if choice[i] {
			b[i/8] |= 1 << uint(i%8)
		}
	}
	return adler32.Checksum(b)
}

// Returns if the given list contains the given element
func listContains(list *list.List, individual *oo.Individual) bool {
	for e := list.Front(); e != nil; e = e.Next() {
		i := e.Value.(*oo.Individual)
		if i == individual {
			return true
		}
	}
	return false
}

func getTime(startTime time.Time) string {
	currentTime := time.Now()
	secondsElapsed := currentTime.Unix() - startTime.Unix()

	addZero := ""
	if secondsElapsed%60 < 10 {
		addZero = "0"
	}
	return strconv.FormatInt(secondsElapsed/60, 10) + ":" + addZero + strconv.FormatInt(secondsElapsed%60, 10)
}
