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

package ants

import (
	"fmt"
	"log"
	"ls"
	"math"
	"math/rand"
	"oo"
	"strconv"
	"time"
)

const (
	ants        = 20
	alpha       = 1.0 // pheromone
	beta        = 2.0 // heuristic
	pBest       = 0.5
	persistence = 0.9 // usually in [0.7, 0.99]

	initialSolutionWeight = 100
)

type antSystem struct {
	file              *oo.File
	rnd               *rand.Rand
	tauMin            float64
	tauMax            float64
	pheromones        [][]float64
	maxLength         int
	maxCompressivness float64
}

func printConst() {
	log.Printf("ants=%v\n", ants)
	log.Printf("alpha=%v\n", alpha)
	log.Printf("beta=%v\n", beta)
	log.Printf("pBest=%v\n", pBest)
	log.Printf("persistence=%v\n", persistence)
	log.Printf("initialSolutionWeight=%v\n", initialSolutionWeight)
}

func (as *antSystem) setTauMin() {
	n := float64(as.file.Size)
	nominator := as.tauMax * (1 - math.Pow(pBest, 1/n))
	denominator := (n/2 - 1) * math.Pow(pBest, 1/n)
	as.tauMin = nominator / denominator
}

func (as *antSystem) setMaxCompressivness() {
	as.maxLength = -1
	as.maxCompressivness = -1
	for _, nt := range as.file.NonTerminals {
		if nt.Length > as.maxLength {
			as.maxLength = nt.Length
		}
		comp := float64((nt.Length-1)*(nt.Count-1) - 2)
		if comp > as.maxCompressivness {
			as.maxCompressivness = comp
		}
	}
}

func (as *antSystem) heuristic(i, j int) float64 {
	if j == len(as.file.Graph[i]) {
		return 1 / as.maxCompressivness
	}
	ntId := as.file.Graph[i][j].Id
	nt := as.file.NonTerminals[ntId]
	comp := (nt.Length-1)*(nt.Count-1) - 2
	return float64(comp) / as.maxCompressivness
}

// Set all pheremone values to tau_max
func (as *antSystem) initPheromones() {
	as.pheromones = make([][]float64, len(as.file.Graph))
	for i := range as.pheromones {
		as.pheromones[i] = make([]float64, len(as.file.Graph[i])+1)
		for j := range as.pheromones[i] {
			as.pheromones[i][j] = as.tauMax
		}
	}
}

func (as *antSystem) updatePheromones(path []int, score int) {
	// reduce pheromone on all edges
	for i := range as.pheromones {
		for j := range as.pheromones[i] {
			as.pheromones[i][j] = math.Max(as.tauMin, persistence*as.pheromones[i][j])
		}
	}

	// add pheromone to given solution
	additionalPheromone := 1 / float64(score)
	for i := range as.pheromones {
		j := path[i]
		if j >= 0 {
			as.pheromones[i][j] = math.Max(as.tauMin, as.pheromones[i][j]+additionalPheromone)
		}
	}
}

func (as *antSystem) chooseProbability(i int, j int, denominator float64) float64 {
	// small optimization
	if len(as.pheromones[i]) == 1 {
		return 1.0
	}

	nominator := math.Pow(as.pheromones[i][j], alpha) * math.Pow(as.heuristic(i, j), beta)
	return nominator / denominator
}

func (as *antSystem) probabilityDenomiator(i int) float64 {
	// small optimization
	if len(as.pheromones[i]) == 1 {
		return 1.0
	}

	denominator := 0.0
	for j := range as.pheromones[i] {
		denominator += math.Pow(as.pheromones[i][j], alpha) * math.Pow(as.heuristic(i, j), beta)
	}
	if denominator == 0.0 {
		panic("denominator is zero")
	}
	return denominator
}

func (as *antSystem) chooseNext(choice []bool, i int) int {
	denomiator := as.probabilityDenomiator(i)

	p := as.rnd.Float64() // p in [0.0,1.0)
	sum := 0.0            // sums up to 1.0
	for j := range as.pheromones[i] {
		sum += as.chooseProbability(i, j, denomiator)
		if p <= sum {
			return j
		}
	}
	panic("sum of all probabilities is smaller than 1.0")
}

func (as *antSystem) ant() ([]bool, []int, int) {
	choice := make([]bool, as.file.NtCount)
	path := make([]int, as.file.Size)
	for i := range path {
		path[i] = -1
	}

	pos := 0
	for pos < as.file.Size-1 {
		index := as.chooseNext(choice, pos)
		path[pos] = index
		if index == len(as.file.Graph[pos]) {
			// no NT chosen
			pos++
		} else {
			nt := as.file.Graph[pos][index]
			pos += nt.Length
			choice[nt.Id] = true
		}
	}
	score := as.file.Score(choice)
	return choice, path, score
}

func (as *antSystem) antRuns(localSearchActive bool) ([]bool, []int, int) {
	// initial best is the empty set
	bestScore := as.file.Size + 1
	bestChoice := make([]bool, as.file.NtCount)
	bestPath := make([]int, as.file.Size)
	for i := 0; i < ants; i++ {
		currentChoice, currentPath, currentScore := as.ant()

		// Complete currentChoice by computing the MGP on the chosen non-terminals
		currentChoice = as.file.CompleteChoice(currentChoice)
		newScore := as.file.Score(currentChoice)
		currentScore = newScore

		// local search
		if localSearchActive {
			individual := ls.LocalSearch(as.file, currentChoice)
			if individual.Score < currentScore {
				currentChoice = individual.Choice
				currentScore = individual.Score
				currentPath = as.file.Path(individual.Choice)
			}
		}
		if currentScore < bestScore {
			bestChoice, bestPath, bestScore = currentChoice, currentPath, currentScore
		}
	}
	return bestChoice, bestPath, bestScore
}

func globalBestFrequency(iteration int) int {
	// One alternative:
	/*if iteration <= 25 {
		// Nearly never use global best
		return 100
	} else if iteration <= 75 {
		return 5
	} else if iteration <= 125 {
		return 3
	} else if iteration <= 250 {
		return 2
	}*/

	// Always use global best
	return 1
}

func MinMaxAntSystem(file *oo.File, initialSolution *oo.Individual, rnd *rand.Rand, handleSolution func(*oo.File, *oo.Solution), startTime time.Time, localSearchActive, csv bool, number int, hybrid bool) {
	printConst()
	as := &antSystem{
		tauMax: 1 - 1.0/float64(file.Size),
		file:   file,
		rnd:    rnd,
	}
	as.setTauMin()
	as.setMaxCompressivness()
	as.initPheromones()

	// Use initial solution
	globalBestChoice := initialSolution.Choice
	globalBestPath := file.Path(initialSolution.Choice)
	globalBestScore := initialSolution.Score
	for j := 0; j < initialSolutionWeight; j++ {
		as.updatePheromones(globalBestPath, globalBestScore)
	}

	iterationsWoImprovement := 0

	i := 0
	for ; ; i++ {
		currentChoice, currentPath, currentScore := as.antRuns(localSearchActive)
		// Store the best-so-far solution
		if currentScore < globalBestScore {
			globalBestChoice, globalBestPath, globalBestScore = currentChoice, currentPath, currentScore
			handleSolution(file, &oo.Solution{
				Score:   globalBestScore,
				Choice:  globalBestChoice,
				NoScore: false,
			})

			if csv {
				minutes := int(time.Now().Sub(startTime).Minutes())
				// number, iteration, #OO, minutes, current best
				fmt.Printf("%d, %d, %d, %d, %d\n", number, i, *oo.OoCounter+*ls.OoCounter, minutes, globalBestScore)
			}
			iterationsWoImprovement = 0
		} else {
			iterationsWoImprovement++
			if hybrid && iterationsWoImprovement >= 3 {
				return
			}
		}

		score := currentScore
		useGlobal := false
		if i%globalBestFrequency(i) == 0 {
			useGlobal = true
			currentChoice, currentPath, currentScore = globalBestChoice, globalBestPath, globalBestScore
		}
		if !csv {
			log.Printf("Iteration %d, best: %d    %s, current: %d, global: %v, #OO: %d\n", i, globalBestScore, getTime(startTime), score, useGlobal, *oo.OoCounter+*ls.OoCounter)
		}
		as.updatePheromones(currentPath, currentScore)
	}
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
