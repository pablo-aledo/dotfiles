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
	"flag"
	"io/ioutil"
	"log"
)

var specialCharacters = map[byte]bool{
	/* # */ 35: true,
	/* N */ 78: true,
	/* \ */ 92: true}

var (
	FlagFile    string
	FlagSize    int
	FlagGrammar string
)

// Nonterminal stores the start and the end index of a nonterminal
// in the given grammar file
type Nonterminal struct {
	start int
	end   int
}

type Data struct {
	file   []byte
	nts    map[int]Nonterminal
	result []byte
}

func init() {
	flag.StringVar(&FlagFile, "f", "", "input file")
	flag.IntVar(&FlagSize, "s", -1, "grammar size to check")
	flag.StringVar(&FlagGrammar, "g", "",
		"grammar generated with the size (witness)")
}

func main() {
	flag.Parse()

	file, err := ioutil.ReadFile(FlagFile)
	if err != nil {
		log.Fatalf("Error reading file: %v\n", err)
	}

	grammar, err := ioutil.ReadFile(FlagGrammar)
	if err != nil {
		log.Fatalf("Error reading grammar: %v\n", err)
	}
	if len(grammar) > 0 && grammar[0] != 35 /* # */ {
		log.Fatal("The grammar file does not start with " +
			"a nonterminal")
	}

	d := &Data{
		file:   grammar,
		nts:    make(map[int]Nonterminal),
		result: make([]byte, 0),
	}
	startNt, size := d.readNonterminals()
	if startNt == -1 {
		log.Fatal("No nonterminal has been found in " +
			"the grammar file")
	}

	log.Printf("Number of nonterminals: %d", len(d.nts))
	log.Printf("Size of the input grammar: %d", size)

	// Reconstruction of the start nonterminal leads to the
	// reconstruction of the whole text (stored in d.result).
	d.reconstruct(startNt)

	accept := true
	if size != FlagSize {
		log.Printf("Expected grammar size is %d, but was %d.\n", FlagSize, size)
	} else if len(d.result) == len(file) {
		for i, b := range d.result {
			if b != file[i] {
				log.Printf("Character mismatch at index %d.\n", i+1)
				accept = false
				break
			}
		}
	} else {
		log.Printf("File length is %d, but grammar reconstruction yields %d.\n", len(file), len(d.result))
		accept = false
	}

	if accept {
		log.Println("accepts")
	} else {
		log.Println("rejects")
	}
}

// readNonterminals iterates over the input grammar and stores
// the indexes for each nonterminal in a map.
func (d *Data) readNonterminals() (startNt, size int) {
	startNt = -1
	size = 0

	start := 3
	end := -1
	currentNt := -1
	nextEscaped := false
	for i := 0; i < len(d.file); i++ {
		if !nextEscaped {
			switch d.file[i] {
			case 35: /* # */
				// The range of the nonterminal that just
				// ended is stored in the map.
				if currentNt >= 0 {
					end = i - 1
					d.nts[currentNt] = Nonterminal{
						start: start,
						end:   end,
					}
					start = i + 3
				}
				high := d.file[i+1]
				low := d.file[i+2]
				currentNt = int(low) + (int(high) << 8)
				// The first nonterminal is used as the starting
				// point even if its number is not 0.
				if startNt == -1 {
					startNt = currentNt
				}
				i += 2
			case 78: /* N */
				// This case is handled in the reconstruct function.
				i += 2
			case 92: /* \ */
				nextEscaped = true
			}
			// In all the above cases the size increases by one.
			// Escaped characters are only counted once.
			size++
		} else {
			nextEscaped = false
		}
	}
	// The last nonterminal is stored in the map.
	end = len(d.file) - 1
	d.nts[currentNt] = Nonterminal{start: start, end: end}
	return
}

// reconstruct returns the sequence represented by the nonterminal
// with the given number with all references to other nonterminals
// removed.
func (d *Data) reconstruct(ntNum int) {
	nt, present := d.nts[ntNum]
	if !present {
		log.Fatalf("Nonterminal %d does not exist\n", ntNum)
	}

	for i := nt.start; i <= nt.end; i++ {
		switch d.file[i] {
		case 92: /* \ */
			if i+1 < len(d.file) && specialCharacters[d.file[i+1]] {
				d.result = append(d.result, d.file[i+1])
				i++
			} else {
				log.Fatalf("Unexpected character after "+
					"backslash: %v\n", d.file[i+1])
			}
		case 78: /* N */
			high := d.file[i+1]
			low := d.file[i+2]
			currentNt := int(low) + (int(high) << 8)
			d.reconstruct(currentNt)
			i += 2
		default:
			d.result = append(d.result, d.file[i])
		}
	}
}
