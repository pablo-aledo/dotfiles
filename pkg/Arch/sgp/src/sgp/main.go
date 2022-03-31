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

package main

import (
	"ants"
	"ea"
	"flag"
	"fmt"
	"io/ioutil"
	"irr"
	"log"
	"ls"
	"math/rand"
	"oo"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"time"
)

var (
	// command line flags
	FlagFilename           string
	FlagGrammarName        string
	FlagTimeLimit          time.Duration
	FlagGrammarSizeLimit   int
	FlagCpuProfile         string
	FlagGrammarOutputLimit int
	FlagGrammarNumber      int
	FlagIrrRuns            int
	FlagSeed               int64
	FlagMetaheuristic      string
	FlagLocalSearch        bool
	FlagPrintGrammar       bool
	FlagRandomizedIrr      bool
	FlagCsvOutput          bool
	FlagInitialLs          bool
	FlagLsAddCandidates    int
)

var (
	startTime = time.Now()
	//solutionChannel     = make(chan *oo.Solution, 100)

	bestSolution *oo.Solution
)

type EncodedSolution struct {
	Bytes  []byte
	Choice []bool
	Score  int
}

func init() {
	flag.StringVar(&FlagFilename, "i", "", "input file")
	flag.StringVar(&FlagGrammarName, "choice", "", "file containing an encoded choice of yields to start with")
	flag.DurationVar(&FlagTimeLimit, "timelimit", 0, "time limit with unit (e.g. 10s)")
	flag.IntVar(&FlagGrammarSizeLimit, "sizelimit", -1, "stop if a grammar of this size is reached")
	flag.StringVar(&FlagCpuProfile, "cpuprofile", "", "enables CPU profiling")
	flag.IntVar(&FlagGrammarOutputLimit, "outputlimit", -1, "output every grammar of this or smaller size")
	flag.IntVar(&FlagGrammarNumber, "number", -1, "this number is appended to the created files")
	flag.IntVar(&FlagIrrRuns, "irr", 10, "number of IRR runs")
	flag.Int64Var(&FlagSeed, "seed", -1, "seed for pseudo-random number generator")
	flag.StringVar(&FlagMetaheuristic, "m", "hybrid", "metaheuristic (genetic, ants, hybrid, none)")
	flag.BoolVar(&FlagLocalSearch, "ls", true, "use local search")
	flag.BoolVar(&FlagPrintGrammar, "print", false, "prints the grammar human readable to stdout")
	flag.BoolVar(&FlagRandomizedIrr, "rirr", false, "randomized IRR")
	flag.BoolVar(&FlagCsvOutput, "csv", false, "output in CSV format")
	flag.BoolVar(&FlagInitialLs, "initls", false, "save intermediate results of initial local search")
	flag.IntVar(&FlagLsAddCandidates, "lscandidates", 1000, "max #candidates considered for bottom-up local search")
}

func printFlags() {
	log.Printf("--i=%v\n", FlagFilename)
	log.Printf("--choice=%v\n", FlagGrammarName)
	log.Printf("--timelimit=%v\n", FlagTimeLimit)
	log.Printf("--sizelimit=%v\n", FlagGrammarSizeLimit)
	log.Printf("--cpuprofile=%v\n", FlagCpuProfile)
	log.Printf("--outputlimit=%v\n", FlagGrammarOutputLimit)
	log.Printf("--number=%v\n", FlagGrammarNumber)
	log.Printf("--irr=%v\n", FlagIrrRuns)
	log.Printf("--seed=%v\n", FlagSeed)
	log.Printf("--m=%v\n", FlagMetaheuristic)
	log.Printf("--ls=%v\n", FlagLocalSearch)
	log.Printf("--rirr=%v\n", FlagRandomizedIrr)
	log.Printf("--csv=%v\n", FlagCsvOutput)
	log.Printf("--initls=%v\n", FlagInitialLs)
	log.Printf("--lscandidates=%v\n", FlagLsAddCandidates)
}

func main() {
	flag.Parse()
	printFlags()

	if FlagFilename == "" {
		log.Fatal("An input file has to be provided")
	}
	filename := FlagFilename

	// read file and transform the byte array into an uint32 array
	in, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Error while reading file: ", err)
	}
	data := make([]uint32, len(in), len(in)+10)
	for i := 0; i < len(in); i++ {
		data[i] = uint32(in[i])
	}

	// profile if enabled
	if FlagCpuProfile != "" {
		f, err := os.Create(FlagCpuProfile)
		if err != nil {
			log.Fatalf("Error creating profile: %v\n", err)
		}
		pprof.StartCPUProfile(f)
	}

	// get all repeats/non-terminals and construct the graph for the shortest path calculations
	file := oo.NewFile(filename, data)
	file.Construct()
	// because of the heavy memory usage of file.Construct() a GC run is forced
	runtime.GC()

	maxNtLength := 0
	maxNtCount := 0
	for _, nt := range file.NonTerminals {
		if nt.Length > maxNtLength {
			maxNtLength = nt.Length
		}
		if nt.Count > maxNtCount {
			maxNtCount = nt.Count
		}
	}

	log.Printf("File size: %d\n", file.Size)
	log.Printf("Number of nonterminals: %d (max. length: %d, max. count: %d)\n", file.NtCount, maxNtLength, maxNtCount)

	seed := FlagSeed
	if FlagSeed == -1 {
		seed = time.Now().Unix() + int64(FlagGrammarNumber)*1000
	}
	rnd := rand.New(rand.NewSource(seed))
	log.Printf("Seed %v\n", seed)

	if FlagTimeLimit > 0 {
		go timeoutTracker(file)
	}

	bestSolution = &oo.Solution{
		Score:   -2,
		Choice:  make([]bool, file.NtCount),
		NoScore: true,
	}

	if FlagLocalSearch {
		ls.Init(file, FlagLsAddCandidates)
	}

	best := &oo.Individual{
		Score:  -1,
		Choice: nil,
	}
	if FlagGrammarName != "" {
		log.Printf("Read in grammar from: %s\n", FlagGrammarName)
		// read in the given choice/solution
		choiceEncoded, err := ioutil.ReadFile(FlagGrammarName)
		if err != nil {
			panic(err.Error())
		}
		choice := make([]bool, file.NtCount)

		if len(choiceEncoded) != file.NtCount/8+1 {
			log.Printf("Expected %v but was %v\n", file.NtCount/8+1, len(choiceEncoded))
			log.Fatal("The given grammar file has the wrong size for the given problem")
		}

		// decode the read in choice
		for i, _ := range choice {
			choice[i] = choiceEncoded[i/8]&(1<<uint(i%8)) != 0
		}
		best = &oo.Individual{-1, choice}
		best.Score = file.Score(best.Choice)

		bestSolution.NoScore = false
		bestSolution.Score = best.Score
		copy(bestSolution.Choice, best.Choice)
	} else {

		log.Printf("Generate initial solution with IRR-MC and local search\n")
		// initial IRR-MC
		if FlagRandomizedIrr {
			bestIrrScore, bestIrrChoice := irr.RandomizedIrr(file, rnd, FlagIrrRuns)
			best = &oo.Individual{
				Score:  bestIrrScore,
				Choice: bestIrrChoice,
			}
		} else {
			bestIrrScore, bestIrrChoice := irr.AllIrr(file, FlagIrrRuns)
			best = &oo.Individual{
				Score:  bestIrrScore,
				Choice: bestIrrChoice,
			}
		}
		log.Printf("Best out of IRR + OO: %d\n", best.Score)
		handleSolution(file, &oo.Solution{
			Score:   best.Score,
			Choice:  best.Choice,
			NoScore: false,
		})

		if FlagLocalSearch {
			// initial local search
			timeLsStart := time.Now()
			if FlagInitialLs {
				// save intermediate results of initial local search
				best.Choice = ls.LocalSearchDown(file, best.Choice, handleSolution)
			} else {
				best.Choice = ls.LocalSearchDown(file, best.Choice, nil)
			}
			timeLsEnd := time.Now()
			best.Score = file.Score(best.Choice)
			log.Printf("IRR + OO + ls: %d\n", best.Score)
			log.Printf("Ls time: %v\n", timeLsEnd.Sub(timeLsStart))
			handleSolution(file, &oo.Solution{
				Score:   best.Score,
				Choice:  best.Choice,
				NoScore: false,
			})
		}
	}

	switch FlagMetaheuristic {
	case "genetic":
		ea.Run(file, best, rnd, handleSolution, startTime, FlagLocalSearch, FlagCsvOutput, FlagGrammarNumber)
	case "ants":
		ants.MinMaxAntSystem(file, best, rnd, handleSolution, startTime, FlagLocalSearch, FlagCsvOutput, FlagGrammarNumber, false /* hybrid */)
	case "hybrid":
		ants.MinMaxAntSystem(file, best, rnd, handleSolution, startTime, FlagLocalSearch, FlagCsvOutput, FlagGrammarNumber, true /* hybrid */)
		best = &oo.Individual{
			Score:  bestSolution.Score,
			Choice: bestSolution.Choice,
		}
		ea.Run(file, best, rnd, handleSolution, startTime, FlagLocalSearch, FlagCsvOutput, FlagGrammarNumber)
	case "none":
		// just do nothing
	default:
		log.Printf("No match for the metaheuristic: %s", FlagMetaheuristic)

	}
}

func handleSolution(file *oo.File, s *oo.Solution) {
	if s.NoScore || bestSolution.NoScore || bestSolution.Score > s.Score {
		copy(bestSolution.Choice, s.Choice)
		bestSolution.Score = s.Score
		bestSolution.NoScore = s.NoScore

		if !bestSolution.NoScore && bestSolution.Score <= FlagGrammarSizeLimit {
			saveChoice(file, &EncodedSolution{nil, bestSolution.Choice, bestSolution.Score})
			pprof.StopCPUProfile()
			printGrammar(file, bestSolution.Choice)
			log.Printf("Exit: grammar size of %v in %v, #OO: %d\n", FlagGrammarSizeLimit, time.Now().Sub(startTime), *oo.OoCounter+*ls.OoCounter)
			os.Exit(0)
		}

		if !bestSolution.NoScore && bestSolution.Score <= FlagGrammarOutputLimit {
			saveChoice(file, &EncodedSolution{nil, bestSolution.Choice, bestSolution.Score})
		}
	}
}

func timeoutTracker(file *oo.File) {
	alreadyElapsed := time.Now().Sub(startTime)
	timeLeft := FlagTimeLimit - alreadyElapsed
	<-time.After(timeLeft)

	saveChoice(file, &EncodedSolution{nil, bestSolution.Choice, bestSolution.Score})
	pprof.StopCPUProfile()
	printGrammar(file, bestSolution.Choice)
	log.Printf("Exit: timeout of %v seconds reached, #OO: %d\n", FlagTimeLimit.Seconds(), *oo.OoCounter+*ls.OoCounter)
	os.Exit(0)
}

// Returns a deep copy of the non-terminal choice of the solution
func copyChoice(solution *oo.Solution) []bool {
	choiceCopy := make([]bool, len(solution.Choice))
	copy(choiceCopy, solution.Choice)
	return choiceCopy
}

// Writes the non-terminal choice to a file
func saveChoice(file *oo.File, encodedSolution *EncodedSolution) {
	size := file.NtCount/8 + 1
	choiceEncoded := make([]byte, size)

	// encode choice
	for i := 0; i < file.NtCount; i++ {
		if encodedSolution.Choice[i] {
			choiceEncoded[i/8] |= 1 << uint(i%8)
		}
	}
	now := time.Now()
	numberString := ""
	if FlagGrammarNumber != -1 {
		numberString = "_" + strconv.Itoa(FlagGrammarNumber)
	}
	timestamp := strconv.Itoa(now.Year()) + "_" + padZero(int(now.Month())) + "_" + padZero(now.Day()) + "_" + padZero(now.Hour()) + "_" + padZero(now.Minute())
	baseName := file.Name + "_" + strconv.Itoa(encodedSolution.Score) + numberString + "_" + timestamp
	ioutil.WriteFile(baseName+".choice", choiceEncoded, 0600)

	encodedBytes, _ := file.ScoreWithReconstruction(encodedSolution.Choice)
	ioutil.WriteFile(baseName+".out", encodedBytes, 0600)
}

func padZero(i int) string {
	if i < 10 {
		return "0" + strconv.Itoa(i)
	}
	return strconv.Itoa(i)
}

// Returns the elapsed time as a formated string
func getTime() string {
	currentTime := time.Now()
	secondsElapsed := currentTime.Unix() - startTime.Unix()

	addZero := ""
	if secondsElapsed%60 < 10 {
		addZero = "0"
	}
	return strconv.FormatInt(secondsElapsed/60, 10) + ":" + addZero + strconv.FormatInt(secondsElapsed%60, 10)
}

func printGrammar(file *oo.File, choice []bool) {
	if !FlagPrintGrammar {
		return
	}
	result := file.PlainReconstruction(choice)
	ntCounter := 0
	fmt.Printf("%d -> ", ntCounter)
	for _, r := range result {
		if r == 256 {
			ntCounter++
			fmt.Printf("\n%d -> ", ntCounter)
		} else if r > 256 {
			fmt.Printf("N_{%d}", r-257)
		} else {
			fmt.Printf("%c", byte(r))
		}
	}
	fmt.Printf("\n")
}
