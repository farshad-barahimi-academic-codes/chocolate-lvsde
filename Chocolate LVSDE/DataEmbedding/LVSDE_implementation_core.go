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
	"fmt"
	"github.com/emirpasic/gods/queues/priorityqueue"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

type DataEmbeddingTechniqueLVSDE struct {
	VisualDensityAdjustmentParameter                float64
	NumberOfNeighboursForBuildingNeighbourhoodGraph int32
	NumberOfParallelSlices                          int32
	WaitGroup                                       sync.WaitGroup
	DataAbstractionSet                              *DataAbstraction.DataAbstractionSet
	CurrentPhase                                    int32
	Epsilon                                         float64
	SquaredBaseDistance                             float64
	BaseDistance                                    float64
	PrecomputedSineOfAxisAngle                      [36]float64
	PrecomputedCosineOfAxisAngle                    [36]float64
	OriginalSpaceMaximumTransformedDistance         float64
	VisualSpaceMaximumDistanceFirstIteration        float64
	Iteration                                       int32
	TemperatureAdjustment                           int32
	Temperature                                     float64
	InitialTemperature                              float64
	GrayLayerDataAbstractionUnitCapacity            int32
	GrayLayerDataAbstractionUnitSize                int32
	Width                                           float64
	Height                                          float64
	EmbeddingDetails                                DataAbstraction.EmbeddingDetails
	FrameLowX                                       float64
	FrameHighX                                      float64
	FrameLowY                                       float64
	FrameHighY                                      float64
	RandomSeed                                      int64
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) EmbedData(dataAbstractionSet DataAbstraction.DataAbstractionSet) {

	dataEmbeddingTechniqueLVSDE.CurrentPhase = 1
	dataEmbeddingTechniqueLVSDE.InitialTemperature = 100.0
	dataEmbeddingTechniqueLVSDE.TemperatureAdjustment = -1
	dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity = -1
	dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitSize = 0
	dataEmbeddingTechniqueLVSDE.Width = 1000.0
	dataEmbeddingTechniqueLVSDE.Height = 1000.0
	var numberOfIterations int32 = 1830
	dataEmbeddingTechniqueLVSDE.EmbeddingDetails.EmbeddingIterations = make([][]*DataAbstraction.DataAbstractionUnitVisibility, numberOfIterations)
	dataEmbeddingTechniqueLVSDE.DataAbstractionSet = &dataAbstractionSet

	dataEmbeddingTechniqueLVSDE.PerformStartingCalculation()

	for dataEmbeddingTechniqueLVSDE.Iteration = 1; dataEmbeddingTechniqueLVSDE.Iteration <= numberOfIterations; dataEmbeddingTechniqueLVSDE.Iteration++ {
		if dataEmbeddingTechniqueLVSDE.Iteration%300 == 0 || dataEmbeddingTechniqueLVSDE.Iteration == 1 || dataEmbeddingTechniqueLVSDE.Iteration == numberOfIterations {
			fmt.Printf("LVSDE iteration %04d starting at %s\n", int(dataEmbeddingTechniqueLVSDE.Iteration), time.Now().Format(time.UnixDate))
		}

		dataEmbeddingTechniqueLVSDE.Temperature = dataEmbeddingTechniqueLVSDE.InitialTemperature - (float64(dataEmbeddingTechniqueLVSDE.Iteration-dataEmbeddingTechniqueLVSDE.TemperatureAdjustment)/1000.0)*dataEmbeddingTechniqueLVSDE.InitialTemperature

		var i, j int32
		var numberOfDataAbstractionUnits int32 = int32(len(dataAbstractionSet.DataAbstractionUnits))
		for i = 0; i < numberOfDataAbstractionUnits; i++ {
			for j = 0; j < int32(len(dataAbstractionSet.DataAbstractionUnits[i].TemporaryVisualSpaceCoordinates)); j++ {
				dataAbstractionSet.DataAbstractionUnits[i].TemporaryVisualSpaceCoordinates[j][0] = 0
				dataAbstractionSet.DataAbstractionUnits[i].TemporaryVisualSpaceCoordinates[j][1] = 0

				dataAbstractionSet.DataAbstractionUnits[i].VisualSpacePositiveReplicationPressuresPerAxis[j] = [36]float64{}
				dataAbstractionSet.DataAbstractionUnits[i].VisualSpaceNegativeReplicationPressuresPerAxis[j] = [36]float64{}
			}
		}

		dataEmbeddingTechniqueLVSDE.WaitGroup.Add(int(dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices))
		for i = 0; i < dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices; i++ {
			go dataEmbeddingTechniqueLVSDE.CalculateRepulsiveForcesSlice(i)
		}
		dataEmbeddingTechniqueLVSDE.WaitGroup.Wait()

		dataEmbeddingTechniqueLVSDE.WaitGroup.Add(int(dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices))
		for i = 0; i < dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices; i++ {
			go dataEmbeddingTechniqueLVSDE.CalculateAttractiveForcesSlice1(i)
		}
		dataEmbeddingTechniqueLVSDE.WaitGroup.Wait()

		dataEmbeddingTechniqueLVSDE.WaitGroup.Add(int(dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices))
		for i = 0; i < dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices; i++ {
			go dataEmbeddingTechniqueLVSDE.CalculateAttractiveForcesSlice2(i)
		}
		dataEmbeddingTechniqueLVSDE.WaitGroup.Wait()

		for i = 0; i < numberOfDataAbstractionUnits; i++ {
			dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[i]

			if dataAbstractionUnit.AreAllVisualSpaceProjectionsFrozen {
				continue
			}

			for j = 0; j < int32(len(dataAbstractionUnit.TemporaryVisualSpaceCoordinates)); j++ {
				length := math.Sqrt(math.Pow(dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][0], 2) + math.Pow(dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][1], 2))
				if length < dataEmbeddingTechniqueLVSDE.Temperature {
					dataAbstractionUnit.VisualSpaceCoordinates[j][0] += dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][0]
					dataAbstractionUnit.VisualSpaceCoordinates[j][1] += dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][1]
				} else {
					dataAbstractionUnit.VisualSpaceCoordinates[j][0] += dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][0] * (dataEmbeddingTechniqueLVSDE.Temperature / length)
					dataAbstractionUnit.VisualSpaceCoordinates[j][1] += dataAbstractionUnit.TemporaryVisualSpaceCoordinates[j][1] * (dataEmbeddingTechniqueLVSDE.Temperature / length)
				}

				x := dataAbstractionUnit.VisualSpaceCoordinates[j][0]
				y := dataAbstractionUnit.VisualSpaceCoordinates[j][1]

				if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
					panic("Not finished successfully. Unstable floating point calculations.")
				}

				if dataEmbeddingTechniqueLVSDE.CurrentPhase >= 2 {
					if dataAbstractionUnit.VisualSpaceCoordinates[j][0] < dataEmbeddingTechniqueLVSDE.FrameLowX {
						dataAbstractionUnit.VisualSpaceCoordinates[j][0] = dataEmbeddingTechniqueLVSDE.FrameLowX
					}

					if dataAbstractionUnit.VisualSpaceCoordinates[j][0] > dataEmbeddingTechniqueLVSDE.FrameHighX {
						dataAbstractionUnit.VisualSpaceCoordinates[j][0] = dataEmbeddingTechniqueLVSDE.FrameHighX
					}

					if dataAbstractionUnit.VisualSpaceCoordinates[j][1] < dataEmbeddingTechniqueLVSDE.FrameLowY {
						dataAbstractionUnit.VisualSpaceCoordinates[j][1] = dataEmbeddingTechniqueLVSDE.FrameLowY
					}

					if dataAbstractionUnit.VisualSpaceCoordinates[j][1] > dataEmbeddingTechniqueLVSDE.FrameHighY {
						dataAbstractionUnit.VisualSpaceCoordinates[j][1] = dataEmbeddingTechniqueLVSDE.FrameHighY
					}
				}
			}
		}

		if dataEmbeddingTechniqueLVSDE.CurrentPhase == 2 {
			if dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity == -1 {
				dataEmbeddingTechniqueLVSDE.CalculateGrayLayerCapacity()
			}

			if dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitSize < dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity {
				dataEmbeddingTechniqueLVSDE.MoveAndFreezeOneDataAbstractionUnitToGrayLayer()
			}
		}

		dataEmbeddingTechniqueLVSDE.EmbeddingDetails.EmbeddingIterations[dataEmbeddingTechniqueLVSDE.Iteration-1] = make([]*DataAbstraction.DataAbstractionUnitVisibility, numberOfDataAbstractionUnits)
		for i = 0; i < numberOfDataAbstractionUnits; i++ {
			dataEmbeddingTechniqueLVSDE.EmbeddingDetails.EmbeddingIterations[dataEmbeddingTechniqueLVSDE.Iteration-1][i] = dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i].ToDataAbstractionUnitVisibility(dataEmbeddingTechniqueLVSDE.Iteration, "LVSDE")
		}

		dataEmbeddingTechniqueLVSDE.ChangePhaseIfRequired()
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) CalculateRepulsiveForcesSlice(sliceNumber int32) {
	defer dataEmbeddingTechniqueLVSDE.WaitGroup.Done()

	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i, j, k, l int32
	for i = sliceNumber; i < numberOfDataAbstractionUnits; i += dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices {
		dataAbstractionUnit1 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		for j = 0; j < int32(len(dataAbstractionUnit1.VisualSpaceCoordinates)); j++ {
			visualSpaceCoordinates1 := dataAbstractionUnit1.VisualSpaceCoordinates[j]
			isIneffective1 := dataAbstractionUnit1.AreAllVisualSpaceProjectionsIneffective

			for k = 0; k < numberOfDataAbstractionUnits; k++ {
				dataAbstractionUnit2 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[k]

				for l = 0; l < int32(len(dataAbstractionUnit2.VisualSpaceCoordinates)); l++ {
					visualSpaceCoordinates2 := dataAbstractionUnit2.VisualSpaceCoordinates[l]
					isIneffective2 := dataAbstractionUnit2.AreAllVisualSpaceProjectionsIneffective

					if i == k && j == l {
						continue
					}

					if isIneffective1 || isIneffective2 {
						continue
					}

					horizontalDifference := visualSpaceCoordinates1[0] - visualSpaceCoordinates2[0]
					verticalDifference := visualSpaceCoordinates1[1] - visualSpaceCoordinates2[1]
					visualDistance := math.Sqrt(math.Pow(horizontalDifference, 2) + math.Pow(verticalDifference, 2))

					if visualDistance < dataEmbeddingTechniqueLVSDE.Epsilon {
						visualDistance = dataEmbeddingTechniqueLVSDE.Epsilon
					}

					repulsiveMagnitude := dataEmbeddingTechniqueLVSDE.SquaredBaseDistance / visualDistance
					repulsiveVectorX := repulsiveMagnitude * (horizontalDifference / visualDistance)
					repulsiveVectorY := repulsiveMagnitude * (verticalDifference / visualDistance)

					dataAbstractionUnit1.TemporaryVisualSpaceCoordinates[j][0] += repulsiveVectorX
					dataAbstractionUnit1.TemporaryVisualSpaceCoordinates[j][1] += repulsiveVectorY

					if dataEmbeddingTechniqueLVSDE.CurrentPhase >= 2 {
						for axis := 0; axis < 36; axis++ {
							pressure := dataEmbeddingTechniqueLVSDE.PrecomputedCosineOfAxisAngle[axis] * repulsiveVectorX
							pressure += dataEmbeddingTechniqueLVSDE.PrecomputedSineOfAxisAngle[axis] * repulsiveVectorY
							if pressure > 0 {
								dataAbstractionUnit1.VisualSpacePositiveReplicationPressuresPerAxis[j][axis] += pressure
							} else {
								dataAbstractionUnit1.VisualSpaceNegativeReplicationPressuresPerAxis[j][axis] += -pressure
							}
						}
					}
				}
			}
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) CalculateAttractiveForcesSlice1(sliceNumber int32) {
	defer dataEmbeddingTechniqueLVSDE.WaitGroup.Done()
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i, j, k, l int32
	for i = sliceNumber; i < numberOfDataAbstractionUnits; i += dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices {
		dataAbstractionUnit1 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]
		dataAbstractionUnit1Index := dataAbstractionUnit1.DataAbstractionUnitNumber

		for j = 0; j < int32(len(dataAbstractionUnit1.VisualSpaceCoordinates)); j++ {
			visualSpaceCoordinates1 := dataAbstractionUnit1.VisualSpaceCoordinates[j]
			isIneffective1 := dataAbstractionUnit1.AreAllVisualSpaceProjectionsIneffective

			for k = 0; k < int32(len(dataAbstractionUnit1.NeighbourIndices[j])); k++ {
				dataAbstractionUnit2 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[dataAbstractionUnit1.NeighbourIndices[j][k][0]]
				dataAbstractionUnit2Index := dataAbstractionUnit2.DataAbstractionUnitNumber
				l = dataAbstractionUnit1.NeighbourIndices[j][k][1]

				visualSpaceCoordinates2 := dataAbstractionUnit2.VisualSpaceCoordinates[l]
				isIneffective2 := dataAbstractionUnit2.AreAllVisualSpaceProjectionsIneffective

				if i == dataAbstractionUnit1.NeighbourIndices[j][k][0] && j == l {
					continue
				}

				if isIneffective1 || isIneffective2 {
					continue
				}

				horizontalDifference := visualSpaceCoordinates1[0] - visualSpaceCoordinates2[0]
				verticalDifference := visualSpaceCoordinates1[1] - visualSpaceCoordinates2[1]
				visualDistance := math.Sqrt(math.Pow(horizontalDifference, 2) + math.Pow(verticalDifference, 2))

				if visualDistance < dataEmbeddingTechniqueLVSDE.Epsilon {
					visualDistance = dataEmbeddingTechniqueLVSDE.Epsilon
				}

				originalSpaceTransformedDistance := dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[dataAbstractionUnit1Index][dataAbstractionUnit2Index]

				attractiveMagnitude1 := visualDistance / dataEmbeddingTechniqueLVSDE.BaseDistance
				attractiveMagnitude1 = math.Pow(attractiveMagnitude1, 1-dataEmbeddingTechniqueLVSDE.VisualDensityAdjustmentParameter)

				attractiveMagnitude2 := originalSpaceTransformedDistance / dataEmbeddingTechniqueLVSDE.OriginalSpaceMaximumTransformedDistance
				attractiveMagnitude2 -= visualDistance / dataEmbeddingTechniqueLVSDE.VisualSpaceMaximumDistanceFirstIteration

				if attractiveMagnitude2 > 0 {
					attractiveMagnitude2 = math.Min(attractiveMagnitude2, math.Abs(attractiveMagnitude1)*0.5)
				} else {
					attractiveMagnitude2 = math.Max(attractiveMagnitude2, math.Abs(attractiveMagnitude1)*(-0.5))
				}

				attractiveMagnitude := attractiveMagnitude1 + attractiveMagnitude2

				attractiveVectorX := -attractiveMagnitude * (horizontalDifference / visualDistance)
				attractiveVectorY := -attractiveMagnitude * (verticalDifference / visualDistance)

				dataAbstractionUnit1.TemporaryVisualSpaceCoordinates[j][0] += attractiveVectorX / dataAbstractionUnit1.Mass[j]
				dataAbstractionUnit1.TemporaryVisualSpaceCoordinates[j][1] += attractiveVectorY / dataAbstractionUnit1.Mass[j]

				if dataEmbeddingTechniqueLVSDE.CurrentPhase >= 2 {
					for axis := 0; axis < 36; axis++ {
						pressure := dataEmbeddingTechniqueLVSDE.PrecomputedCosineOfAxisAngle[axis] * attractiveVectorX
						pressure += dataEmbeddingTechniqueLVSDE.PrecomputedSineOfAxisAngle[axis] * attractiveVectorY
						if pressure > 0 {
							dataAbstractionUnit1.VisualSpacePositiveReplicationPressuresPerAxis[j][axis] += pressure
						} else {
							dataAbstractionUnit1.VisualSpaceNegativeReplicationPressuresPerAxis[j][axis] += -pressure
						}
					}
				}

			}
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) CalculateAttractiveForcesSlice2(sliceNumber int32) {
	defer dataEmbeddingTechniqueLVSDE.WaitGroup.Done()
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i, j, k, l int32
	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit1 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]
		dataAbstractionUnit1Index := dataAbstractionUnit1.DataAbstractionUnitNumber

		for j = 0; j < int32(len(dataAbstractionUnit1.VisualSpaceCoordinates)); j++ {
			visualSpaceCoordinates1 := dataAbstractionUnit1.VisualSpaceCoordinates[j]
			isIneffective1 := dataAbstractionUnit1.AreAllVisualSpaceProjectionsIneffective

			for k = 0; k < int32(len(dataAbstractionUnit1.NeighbourIndices[j])); k++ {
				dataAbstractionUnit2 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[dataAbstractionUnit1.NeighbourIndices[j][k][0]]
				dataAbstractionUnit2Index := dataAbstractionUnit2.DataAbstractionUnitNumber

				if dataAbstractionUnit2Index%dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices != sliceNumber {
					continue
				}

				l = dataAbstractionUnit1.NeighbourIndices[j][k][1]

				visualSpaceCoordinates2 := dataAbstractionUnit2.VisualSpaceCoordinates[l]
				isIneffective2 := dataAbstractionUnit2.AreAllVisualSpaceProjectionsIneffective

				if i == dataAbstractionUnit1.NeighbourIndices[j][k][0] && j == l {
					continue
				}

				if isIneffective1 || isIneffective2 {
					continue
				}

				horizontalDifference := visualSpaceCoordinates1[0] - visualSpaceCoordinates2[0]
				verticalDifference := visualSpaceCoordinates1[1] - visualSpaceCoordinates2[1]
				visualDistance := math.Sqrt(math.Pow(horizontalDifference, 2) + math.Pow(verticalDifference, 2))

				if visualDistance < dataEmbeddingTechniqueLVSDE.Epsilon {
					visualDistance = dataEmbeddingTechniqueLVSDE.Epsilon
				}

				originalSpaceTransformedDistance := dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[dataAbstractionUnit1Index][dataAbstractionUnit2Index]

				attractiveMagnitude1 := visualDistance / dataEmbeddingTechniqueLVSDE.BaseDistance
				attractiveMagnitude1 = math.Pow(attractiveMagnitude1, 1-dataEmbeddingTechniqueLVSDE.VisualDensityAdjustmentParameter)

				attractiveMagnitude2 := originalSpaceTransformedDistance / dataEmbeddingTechniqueLVSDE.OriginalSpaceMaximumTransformedDistance
				attractiveMagnitude2 -= visualDistance / dataEmbeddingTechniqueLVSDE.VisualSpaceMaximumDistanceFirstIteration

				if attractiveMagnitude2 > 0 {
					attractiveMagnitude2 = math.Min(attractiveMagnitude2, math.Abs(attractiveMagnitude1)*0.5)
				} else {
					attractiveMagnitude2 = math.Max(attractiveMagnitude2, math.Abs(attractiveMagnitude1)*(-0.5))
				}

				attractiveMagnitude := attractiveMagnitude1 + attractiveMagnitude2

				attractiveVectorX := attractiveMagnitude * (horizontalDifference / visualDistance)
				attractiveVectorY := attractiveMagnitude * (verticalDifference / visualDistance)

				dataAbstractionUnit2.TemporaryVisualSpaceCoordinates[l][0] += attractiveVectorX / dataAbstractionUnit2.Mass[l]
				dataAbstractionUnit2.TemporaryVisualSpaceCoordinates[l][1] += attractiveVectorY / dataAbstractionUnit2.Mass[l]

				if dataEmbeddingTechniqueLVSDE.CurrentPhase >= 2 {
					for axis := 0; axis < 36; axis++ {
						pressure := dataEmbeddingTechniqueLVSDE.PrecomputedCosineOfAxisAngle[axis] * attractiveVectorX
						pressure += dataEmbeddingTechniqueLVSDE.PrecomputedSineOfAxisAngle[axis] * attractiveVectorY
						if pressure > 0 {
							dataAbstractionUnit2.VisualSpacePositiveReplicationPressuresPerAxis[l][axis] += pressure
						} else {
							dataAbstractionUnit2.VisualSpaceNegativeReplicationPressuresPerAxis[l][axis] += -pressure
						}
					}
				}
			}
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) ChangePhaseIfRequired() {
	if dataEmbeddingTechniqueLVSDE.Iteration == 500 {
		dataEmbeddingTechniqueLVSDE.CurrentPhase = 2

		dataAbstractionUnits := dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits
		var numberOfDataAbstractionUnits int32 = int32(len(dataAbstractionUnits))
		var i, j int32

		var xLow float64 = math.Inf(1)
		var xHigh float64 = math.Inf(-1)

		var yLow float64 = math.Inf(1)
		var yHigh float64 = math.Inf(-1)

		for i = 0; i < numberOfDataAbstractionUnits; i++ {
			dataAbstractionUnit := &dataAbstractionUnits[i]

			for j = 0; j < int32(len(dataAbstractionUnit.VisualSpaceCoordinates)); j++ {
				x := dataAbstractionUnit.VisualSpaceCoordinates[j][0]
				y := dataAbstractionUnit.VisualSpaceCoordinates[j][1]

				xLow = math.Min(xLow, x)
				xHigh = math.Max(xHigh, x)
				yLow = math.Min(yLow, y)
				yHigh = math.Max(yHigh, y)
			}
		}

		dataEmbeddingTechniqueLVSDE.FrameLowX = xLow - (xHigh-xLow)/20.0
		dataEmbeddingTechniqueLVSDE.FrameHighX = xHigh + (xHigh-xLow)/20.0
		dataEmbeddingTechniqueLVSDE.FrameLowY = yLow - (yHigh-yLow)/20.0
		dataEmbeddingTechniqueLVSDE.FrameHighY = yHigh + (yHigh-yLow)/20.0

	} else if dataEmbeddingTechniqueLVSDE.Iteration == 950 {
		dataEmbeddingTechniqueLVSDE.CurrentPhase = 3
		dataEmbeddingTechniqueLVSDE.TemperatureAdjustment = 440
		dataEmbeddingTechniqueLVSDE.UnfreezeAndMarkEffectiveGrayLayer()
		dataEmbeddingTechniqueLVSDE.FreezeRedLayer()
	} else if dataEmbeddingTechniqueLVSDE.Iteration == 1340 {
		dataEmbeddingTechniqueLVSDE.CurrentPhase = 4
		dataEmbeddingTechniqueLVSDE.TemperatureAdjustment = 830
		dataEmbeddingTechniqueLVSDE.SplitVerticesOfGrayLayerIfPossible()
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) PerformStartingCalculation() {
	dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices = int32(runtime.NumCPU()) - 1
	fmt.Println("Number of parallel goroutines:", dataEmbeddingTechniqueLVSDE.NumberOfParallelSlices)
	dataEmbeddingTechniqueLVSDE.DataAbstractionSet.ComputeDistancesAfterTransformation()

	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i, j int32

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		originalSpaceTransformedDistanceCompare := func(a, b interface{}) int {
			indexA := a.(*DataAbstraction.DataAbstractionUnit).DataAbstractionUnitNumber
			indexB := b.(*DataAbstraction.DataAbstractionUnit).DataAbstractionUnitNumber
			if dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[i][indexA] < dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[i][indexB] {
				return -1
			} else if dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[i][indexA] == dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[i][indexB] {
				return 0
			} else {
				return 1
			}
		}

		queue := priorityqueue.NewWith(originalSpaceTransformedDistanceCompare)

		for j = 0; j < numberOfDataAbstractionUnits; j++ {
			if i == j {
				continue
			}
			dataAbstractionUnit2 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[j]
			queue.Enqueue(dataAbstractionUnit2)
		}

		dataAbstractionUnit.NeighbourIndices = make([][][2]int32, 1)
		dataAbstractionUnit.NeighbourIndices[0] = make([][2]int32, dataEmbeddingTechniqueLVSDE.NumberOfNeighboursForBuildingNeighbourhoodGraph)
		for j = 0; j < dataEmbeddingTechniqueLVSDE.NumberOfNeighboursForBuildingNeighbourhoodGraph; j++ {
			dataAbstractionUnit2, _ := queue.Dequeue()
			if dataAbstractionUnit == nil {
				panic("Not finished successfully.")
			}

			neighbourIndex := dataAbstractionUnit2.(*DataAbstraction.DataAbstractionUnit).DataAbstractionUnitNumber
			dataAbstractionUnit.NeighbourIndices[0][j][0] = neighbourIndex
			dataAbstractionUnit.NeighbourIndices[0][j][1] = 0
		}
	}

	randomSource := rand.NewSource(dataEmbeddingTechniqueLVSDE.RandomSeed)
	randomGenerator := rand.New(randomSource)
	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]
		dataAbstractionUnit.VisualSpaceCoordinates[0][0] = randomGenerator.Float64() * dataEmbeddingTechniqueLVSDE.Width
		dataAbstractionUnit.VisualSpaceCoordinates[0][1] = randomGenerator.Float64() * dataEmbeddingTechniqueLVSDE.Height
		dataAbstractionUnit.TemporaryVisualSpaceCoordinates[0][0] = 0
		dataAbstractionUnit.TemporaryVisualSpaceCoordinates[0][0] = 0
	}

	dataEmbeddingTechniqueLVSDE.Epsilon = 1e-10
	dataEmbeddingTechniqueLVSDE.SquaredBaseDistance = (dataEmbeddingTechniqueLVSDE.Width * dataEmbeddingTechniqueLVSDE.Height) / float64(numberOfDataAbstractionUnits)
	dataEmbeddingTechniqueLVSDE.BaseDistance = math.Sqrt(dataEmbeddingTechniqueLVSDE.SquaredBaseDistance)

	dataEmbeddingTechniqueLVSDE.OriginalSpaceMaximumTransformedDistance = 0.0
	dataEmbeddingTechniqueLVSDE.VisualSpaceMaximumDistanceFirstIteration = 0.0

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		for j = 0; j < numberOfDataAbstractionUnits; j++ {
			visualSpaceCoordinates1 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i].VisualSpaceCoordinates[0]
			visualSpaceCoordinates2 := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[j].VisualSpaceCoordinates[0]

			horizontalDifference := visualSpaceCoordinates1[0] - visualSpaceCoordinates2[0]
			verticalDifference := visualSpaceCoordinates1[1] - visualSpaceCoordinates2[1]
			visualDistance := math.Sqrt(math.Pow(horizontalDifference, 2) + math.Pow(verticalDifference, 2))

			dataEmbeddingTechniqueLVSDE.OriginalSpaceMaximumTransformedDistance = math.Max(dataEmbeddingTechniqueLVSDE.OriginalSpaceMaximumTransformedDistance, dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DistancesAfterTransformation[i][j])
			dataEmbeddingTechniqueLVSDE.VisualSpaceMaximumDistanceFirstIteration = math.Max(dataEmbeddingTechniqueLVSDE.VisualSpaceMaximumDistanceFirstIteration, visualDistance)

		}
	}

	for axis := 0; axis < 36; axis++ {
		dataEmbeddingTechniqueLVSDE.PrecomputedCosineOfAxisAngle[axis] = math.Cos(math.Pi * float64(axis) * 10.0 / 180.0)
		dataEmbeddingTechniqueLVSDE.PrecomputedSineOfAxisAngle[axis] = math.Sin(math.Pi * float64(axis) * 10.0 / 180.0)
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) SplitVerticesOfGrayLayerIfPossible() {
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i int32

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		if !dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer {
			dataEmbeddingTechniqueLVSDE.SplitVertex(dataAbstractionUnit, 0)
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) UnfreezeAndMarkEffectiveGrayLayer() {
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i int32

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		if !dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer {
			dataAbstractionUnit.AreAllVisualSpaceProjectionsIneffective = false
			dataAbstractionUnit.AreAllVisualSpaceProjectionsFrozen = false
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) FreezeRedLayer() {
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i int32

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		if dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer {
			dataAbstractionUnit.AreAllVisualSpaceProjectionsFrozen = true
		}
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) CalculateGrayLayerCapacity() {
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i int32

	var pressureMean float64 = 0

	var numberOfVisualSpaceProjections int32 = 0

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]
		var maximumPressure float64 = 0
		for axis := 0; axis < 36; axis++ {
			pressure := dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[0][axis] + dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[0][axis]
			maximumPressure = math.Max(maximumPressure, pressure)
		}

		pressureMean += maximumPressure
		numberOfVisualSpaceProjections++

	}

	pressureMean /= float64(numberOfVisualSpaceProjections)

	var pressureStandardDeviation float64 = 0

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		var maximumPressure float64 = 0
		for axis := 0; axis < 36; axis++ {
			pressure := dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[0][axis] + dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[0][axis]
			maximumPressure = math.Max(maximumPressure, pressure)
		}

		pressureStandardDeviation += math.Pow(maximumPressure-pressureMean, 2)
	}

	pressureStandardDeviation /= float64(numberOfVisualSpaceProjections)
	pressureStandardDeviation = math.Sqrt(pressureStandardDeviation)

	dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity = 0

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]

		var maximumPressure float64 = 0
		for axis := 0; axis < 36; axis++ {
			pressure := dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[0][axis] + dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[0][axis]
			maximumPressure = math.Max(maximumPressure, pressure)
		}

		if math.Abs(maximumPressure-pressureMean) > 1.2*pressureStandardDeviation {
			dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity++
		}
	}

	if numberOfDataAbstractionUnits/4 < dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity {
		dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitCapacity = numberOfDataAbstractionUnits / 4
	}
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) MoveAndFreezeOneDataAbstractionUnitToGrayLayer() {
	var numberOfDataAbstractionUnits int32 = int32(len(dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits))
	var i, indexMaximum int32

	var maximumPressureAll float64 = -1.0
	indexMaximum = -1

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnit := &dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[i]
		if dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer == false {
			continue
		}

		for axis := 0; axis < 36; axis++ {
			pressure := dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[0][axis] + dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[0][axis]
			if pressure > maximumPressureAll {
				maximumPressureAll = pressure
				indexMaximum = i
			}
		}
	}

	dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[indexMaximum].AreAllVisualSpaceProjectionsInRedLayer = false
	dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[indexMaximum].AreAllVisualSpaceProjectionsIneffective = true
	dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[indexMaximum].AreAllVisualSpaceProjectionsFrozen = true
	dataEmbeddingTechniqueLVSDE.GrayLayerDataAbstractionUnitSize++
}

func (dataEmbeddingTechniqueLVSDE *DataEmbeddingTechniqueLVSDE) SplitVertex(dataAbstractionUnit *DataAbstraction.DataAbstractionUnit, index int32) {
	var maximumPressure float64 = -1
	hasVertexSplitFailed := true
	selectedAxis := -1
	for axis := 0; axis < 36; axis++ {
		axisPressure := dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[index][axis] + dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[index][axis]
		if axisPressure > maximumPressure {
			maximumPressure = axisPressure
			selectedAxis = axis
			hasVertexSplitFailed = false
		}
	}

	if hasVertexSplitFailed {
		dataAbstractionUnit.HasVertexSplitFailed = true
		return
	}

	preSplitNumberOfNeighbours := len(dataAbstractionUnit.NeighbourIndices[index])

	visualNeighboursIndices1 := make([][2]int32, 0)
	visualNeighboursIndices2 := make([][2]int32, 0)

	for i := 0; i < preSplitNumberOfNeighbours; i++ {
		neighbourOriginalSpaceIndex := dataAbstractionUnit.NeighbourIndices[index][i][0]
		neighbourVisualSpaceIndex := dataAbstractionUnit.NeighbourIndices[index][i][1]
		neighbourDataAbstractionUnit := dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[neighbourOriginalSpaceIndex]

		horizontalDifference := neighbourDataAbstractionUnit.VisualSpaceCoordinates[neighbourVisualSpaceIndex][0] - dataAbstractionUnit.VisualSpaceCoordinates[neighbourVisualSpaceIndex][0]
		verticalDifference := neighbourDataAbstractionUnit.VisualSpaceCoordinates[neighbourVisualSpaceIndex][1] - dataAbstractionUnit.VisualSpaceCoordinates[neighbourVisualSpaceIndex][1]

		indicatorBasedOnCausedPressureOnSelectedAxis := dataEmbeddingTechniqueLVSDE.PrecomputedCosineOfAxisAngle[selectedAxis] * horizontalDifference
		indicatorBasedOnCausedPressureOnSelectedAxis += dataEmbeddingTechniqueLVSDE.PrecomputedSineOfAxisAngle[selectedAxis] * verticalDifference
		if indicatorBasedOnCausedPressureOnSelectedAxis < 0 {
			visualNeighboursIndices1 = append(visualNeighboursIndices1, [2]int32{neighbourOriginalSpaceIndex, neighbourVisualSpaceIndex})
		} else {
			visualNeighboursIndices2 = append(visualNeighboursIndices2, [2]int32{neighbourOriginalSpaceIndex, neighbourVisualSpaceIndex})
		}
	}

	if len(visualNeighboursIndices1) == 0 || len(visualNeighboursIndices2) == 0 {
		dataAbstractionUnit.HasVertexSplitFailed = true
		return
	}

	dataAbstractionUnit.VisualSpaceCoordinates = append(dataAbstractionUnit.VisualSpaceCoordinates, [2]float64{0, 0})
	dataAbstractionUnit.TemporaryVisualSpaceCoordinates = append(dataAbstractionUnit.TemporaryVisualSpaceCoordinates, [2]float64{0, 0})
	dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis = append(dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis, [36]float64{})
	dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis = append(dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis, [36]float64{})
	dataAbstractionUnit.Mass = append(dataAbstractionUnit.Mass, 0)

	dataAbstractionUnit.NeighbourIndices[0] = visualNeighboursIndices1
	dataAbstractionUnit.NeighbourIndices = append(dataAbstractionUnit.NeighbourIndices, visualNeighboursIndices2)

	dataAbstractionUnit.Mass[1] = dataAbstractionUnit.Mass[0] * (float64(len(visualNeighboursIndices2)) / float64(preSplitNumberOfNeighbours))
	dataAbstractionUnit.Mass[0] = dataAbstractionUnit.Mass[0] * (float64(len(visualNeighboursIndices1)) / float64(preSplitNumberOfNeighbours))

	var x, y float64 = 0, 0

	for i := 0; i < len(visualNeighboursIndices2); i++ {
		x += dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[visualNeighboursIndices2[i][0]].VisualSpaceCoordinates[visualNeighboursIndices2[i][1]][0]
		y += dataEmbeddingTechniqueLVSDE.DataAbstractionSet.DataAbstractionUnits[visualNeighboursIndices2[i][0]].VisualSpaceCoordinates[visualNeighboursIndices2[i][1]][1]
	}

	x /= float64(len(visualNeighboursIndices2))
	y /= float64(len(visualNeighboursIndices2))
	dataAbstractionUnit.VisualSpaceCoordinates[1][0] = x
	dataAbstractionUnit.VisualSpaceCoordinates[1][1] = y
}
