/*
	Named "Chocolate LVSDE", this project (including but not limited to this file) is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.
	LVSDE stands for Layered Vertex Splitting Data Embedding.
	LVSDE dimensionality reduction technique (algorithm) was first named Red Gray Plus (subject to possibly some changes or corrections). Red Gray Plus is described in the arXiv preprint numbered 2101.06224 which should be updated soon (not yet) to use the name LVSDE instead.

	The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

	Copyright notice for this code (this implementation of LVSDE) and this file:
	Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.

	All codes in this project including but not limited to this file are written by Farshad Barahimi.

	The purpose of writing this code is academic.
*/

package DataEmbedding

import (
	"github.com/emirpasic/gods/queues/priorityqueue"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"math"
	"sort"
)

func EvaluateEmbedding(dataAbstractionUnitVisibilities []*DataAbstraction.DataAbstractionUnitVisibility, numberOfNeighbours int, evaluationLayers []string, evaluationNeighboursLayers []string) float64 {
	numberOfDataAbstractionUnits := len(dataAbstractionUnitVisibilities)
	var corrects int = 0
	var incorrects = 0

	inEvaluationLayers := make(map[string]bool)
	for _, layer := range evaluationLayers {
		inEvaluationLayers[layer] = true
	}

	inEvaluationNeighboursLayers := make(map[string]bool)
	for _, layer := range evaluationNeighboursLayers {
		inEvaluationNeighboursLayers[layer] = true
	}

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
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
						panic("Not finished successfully.")
					}
					neighbourIndex := a.([]int)[0]
					if neighbourIndex != i {
						counts[int(dataAbstractionUnitVisibilities[neighbourIndex].ClassLabelNumber)]++
					}
				}
			}

			classLabelNumbers := make([]int, 0, len(counts))
			for classLabelNumber := range counts {
				classLabelNumbers = append(classLabelNumbers, classLabelNumber)
			}
			sort.Ints(classLabelNumbers)

			var maximumCount int = -1
			var maximumClassLabelNumber int = -1

			for i := 0; i < len(classLabelNumbers); i++ {
				classLabelNumber := classLabelNumbers[i]
				if counts[classLabelNumber] > maximumCount {
					maximumCount = counts[classLabelNumber]
					maximumClassLabelNumber = classLabelNumber
				}
			}

			if maximumClassLabelNumber == -1 {
				panic("Not finished successfully.")
			}

			if maximumClassLabelNumber == int(dataAbstractionUnitVisibilities[i].ClassLabelNumber) {
				corrects++
			} else {
				incorrects++
			}
		}
	}

	return 100.0 * (float64(corrects) / float64(corrects+incorrects))
}
