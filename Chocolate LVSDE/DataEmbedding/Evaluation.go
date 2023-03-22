/*
	Named "Chocolate LVSDE", this project (including but not limited to this file) is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.

	The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

	Copyright notice for this code (this implementation of LVSDE) and this file:
	Copyright (c) 2022-2023 Farshad Barahimi. Licensed under the MIT license.

	All codes in this project including but not limited to this file are written by Farshad Barahimi.

	The purpose of writing this code is academic.

	LVSDE stands for Layered Vertex Splitting Data Embedding.
	For more information about LVSDE dimensionality reduction technique (algorithm) look at the following arXiv preprint:
	Farshad Barahimi, "Multi-point dimensionality reduction to improve projection layout reliability",  arXiv:2101.06224v3, 2022.
*/

package DataEmbedding

import (
	"github.com/emirpasic/gods/queues/priorityqueue"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

func EvaluateEmbedding(dataAbstractionUnitVisibilitiesToBeShuffled []*DataAbstraction.DataAbstractionUnitVisibility, numberOfNeighbours int, evaluationLayers []string, evaluationNeighboursLayers []string, precision int) []string {
	numberOfDataAbstractionUnits := len(dataAbstractionUnitVisibilitiesToBeShuffled)
	dataAbstractionUnitVisibilities := make([]*DataAbstraction.DataAbstractionUnitVisibility, numberOfDataAbstractionUnits)
	copy(dataAbstractionUnitVisibilities, dataAbstractionUnitVisibilitiesToBeShuffled)

	randomGenerator := rand.New(rand.NewSource(849662123548415231))
	randomGenerator.Shuffle(len(dataAbstractionUnitVisibilities), func(i, j int) {
		dataAbstractionUnitVisibilities[i], dataAbstractionUnitVisibilities[j] = dataAbstractionUnitVisibilities[j], dataAbstractionUnitVisibilities[i]
	})

	var corrects int = 0
	var incorrects = 0
	confusionMatrix := make(map[int]map[int]int)
	var maximumClassLabelNumber = -1

	inEvaluationLayers := make(map[string]bool)
	for _, layer := range evaluationLayers {
		inEvaluationLayers[layer] = true
	}

	inEvaluationNeighboursLayers := make(map[string]bool)
	for _, layer := range evaluationNeighboursLayers {
		inEvaluationNeighboursLayers[layer] = true
	}

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		if int(dataAbstractionUnitVisibilities[i].ClassLabelNumber) > maximumClassLabelNumber {
			maximumClassLabelNumber = int(dataAbstractionUnitVisibilities[i].ClassLabelNumber)
		}

		if inEvaluationLayers[dataAbstractionUnitVisibilities[i].Layer] {
			counts := make(map[int]int)

			for j := 0; j < len(dataAbstractionUnitVisibilities[i].VisualSpaceCoordinates); j++ {
				visualSpaceDistanceCompare := func(a, b interface{}) int {
					A := a.([]int)
					B := b.([]int)

					x := dataAbstractionUnitVisibilities[i].VisualSpaceCoordinates[j][0]
					y := dataAbstractionUnitVisibilities[i].VisualSpaceCoordinates[j][1]
					xA := dataAbstractionUnitVisibilities[A[0]].VisualSpaceCoordinates[A[1]][0]
					yA := dataAbstractionUnitVisibilities[A[0]].VisualSpaceCoordinates[A[1]][1]
					xB := dataAbstractionUnitVisibilities[B[0]].VisualSpaceCoordinates[B[1]][0]
					yB := dataAbstractionUnitVisibilities[B[0]].VisualSpaceCoordinates[B[1]][1]

					distanceA := math.Sqrt(math.Pow(xA-x, 2) + math.Pow(yA-y, 2))
					distanceB := math.Sqrt(math.Pow(xB-x, 2) + math.Pow(yB-y, 2))

					if distanceA < distanceB {
						return -1
					} else if distanceA == distanceB {
						return 0
					} else {
						return 1
					}
				}

				queue := priorityqueue.NewWith(visualSpaceDistanceCompare)

				for t := 0; t < numberOfDataAbstractionUnits; t++ {
					if inEvaluationNeighboursLayers[dataAbstractionUnitVisibilities[t].Layer] {
						for l := 0; l < len(dataAbstractionUnitVisibilities[t].VisualSpaceCoordinates); l++ {
							if i == t && l == j {
								continue
							}

							queue.Enqueue([]int{t, l})
						}
					}
				}

				for t := 0; t < numberOfNeighbours; t++ {
					a, _ := queue.Dequeue()
					if a == nil {
						return []string{"(Not enough neighbours),(Not enough neighbours),(Not enough neighbours)", "(Not enough neighbours)"}
					}
					neighbourIndex := a.([]int)[0]
					if neighbourIndex != i {
						counts[int(dataAbstractionUnitVisibilities[neighbourIndex].ClassLabelNumber)]++
					}
				}
			}

			var maximumOccurrenceCount int = -1
			var maximumOccurrenceClassLabelNumber int = -1

			for i := 0; i <= maximumClassLabelNumber; i++ {
				if counts[i] > maximumOccurrenceCount {
					maximumOccurrenceCount = counts[i]
					maximumOccurrenceClassLabelNumber = i
				}

				if confusionMatrix[i] == nil {
					confusionMatrix[i] = make(map[int]int)
				}
			}

			if maximumOccurrenceClassLabelNumber == -1 {
				panic("Not finished successfully.")
			}

			if maximumOccurrenceClassLabelNumber == int(dataAbstractionUnitVisibilities[i].ClassLabelNumber) {
				corrects++
			} else {
				incorrects++
			}

			confusionMatrix[int(dataAbstractionUnitVisibilities[i].ClassLabelNumber)][maximumOccurrenceClassLabelNumber]++
		}
	}

	var percent float64 = math.NaN()

	if corrects+incorrects > 0 {
		percent = 100.0 * (float64(corrects) / float64(corrects+incorrects))
	}

	confusionMatrixCSV := strings.Builder{}

	for i := 0; i <= maximumClassLabelNumber; i++ {
		for j := 0; j <= maximumClassLabelNumber; j++ {
			if j != 0 {
				confusionMatrixCSV.WriteString(",")
			}

			confusionMatrixCSV.WriteString(strconv.Itoa(confusionMatrix[i][j]))

		}
		confusionMatrixCSV.WriteString("\r\n")
	}

	return []string{strconv.FormatFloat(percent, 'f', precision, 64) + "%," +
		strconv.Itoa(corrects) + "," + strconv.Itoa(incorrects), confusionMatrixCSV.String()}
}
