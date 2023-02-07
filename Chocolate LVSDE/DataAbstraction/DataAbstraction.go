/*
	Named "Chocolate LVSDE", this project (including but not limited to this file) is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.

	The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

	Copyright notice for this code (this implementation of LVSDE) and this file:
	Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.

	All codes in this project including but not limited to this file are written by Farshad Barahimi.

	The purpose of writing this code is academic.

	LVSDE stands for Layered Vertex Splitting Data Embedding.
	For more information about LVSDE dimensionality reduction technique (algorithm) look at the following arXiv preprint:
	Farshad Barahimi, "Multi-point dimensionality reduction to improve projection layout reliability",  arXiv:2101.06224v3, 2022.
*/

package DataAbstraction

import (
	"github.com/emirpasic/gods/queues/priorityqueue"
	"image"
	"image/color"
	"math"
	"strings"
)

type DataAbstractionUnit struct {
	OriginalSpaceCoordinates                       []float64
	ThirtyDimensionalSpaceCoordinates              []float64
	ClassLabelNumber                               int32
	VisualSpaceCoordinates                         [][2]float64
	VisualSpacePositiveReplicationPressuresPerAxis [][36]float64
	VisualSpaceNegativeReplicationPressuresPerAxis [][36]float64
	TemporaryVisualSpaceCoordinates                [][2]float64
	ImageRGB                                       []uint8
	ImageGrayscale                                 []uint8
	ImageWidth                                     int32
	ImageHeight                                    int32
	DataAbstractionUnitNumber                      int32
	AreAllVisualSpaceProjectionsIneffective        bool
	AreAllVisualSpaceProjectionsInRedLayer         bool
	AreAllVisualSpaceProjectionsFrozen             bool
	Mass                                           []float64
	HasVertexSplitFailed                           bool
	NeighbourIndices                               [][][2]int32
	ComparisonVisualSpaceCoordinates               [][2]float64
}

type DataAbstractionSet struct {
	DataAbstractionUnits                 []DataAbstractionUnit
	DistancesBeforeTransformation        [][]float64
	DistancesAfterTransformation         [][]float64
	DistancesBeforeThirtyDimensionalUMAP [][]float64
}

type DataAbstractionUnitVisibility struct {
	ClassLabelNumber          int32        `json:"class_label_number" bson:"c"`
	VisualSpaceCoordinates    [][2]float64 `json:"visual_space_coordinates" bson:"v"`
	DataAbstractionUnitNumber int32        `json:"data_abstraction_unit_number" bson:"d"`
	Iteration                 int32        `json:"iteration" bson:"i"`
	Layer                     string       `json:"layer" bson:"l"`
}

type EmbeddingDetails struct {
	EmbeddingIterations                             [][]*DataAbstractionUnitVisibility `json:"embedding_iterations" bson:"embedding_iterations"`
	ClassLabels                                     []string                           `json:"class_labels" bson:"class_labels"`
	ColoursList                                     []string                           `json:"colours_list" bson:"colours_list"`
	ImageWidth                                      int32                              `json:"image_width" bson:"image_width"`
	RandomSeed                                      string                             `json:"random_seed" bson:"random_seed"`
	RandomState                                     string                             `json:"random_state" bson:"random_state"`
	PreliminaryToThirtyDimensionsUMAP               string                             `json:"preliminary_to_thirty_dimensions_umap" bson:"preliminary_to_thirty_dimensions_umap"`
	NumberOfInitialDataAbstractionUnits             string                             `json:"number_of_initial_data_abstraction_units" bson:"number_of_initial_data_abstraction_units"`
	NumberOfSecondaryDataAbstractionUnits           string                             `json:"number_of_secondary_data_abstraction_units" bson:"number_of_secondary_data_abstraction_units"`
	VisualDensityAdjustmentParameter                string                             `json:"visual_density_adjustment_parameter" bson:"visual_density_adjustment_parameter"`
	NumberOfNeighboursForBuildingNeighbourhoodGraph string                             `json:"number_of_neighbours_for_building_neighbourhood_graph" bson:"number_of_neighbours_for_building_neighbourhood_graph"`
	EvaluationNeighbourhoodSizes                    []string                           `json:"evaluation_neighbourhood_sizes" bson:"evaluation_neighbourhood_sizes"`
	ImagesRedGreenBlueChannels                      [][]uint8                          `json:"images_red_green_blue_channels" bson:"images_red_green_blue_channels"`
	ImagesGrayscaleSingleChannel                    [][]uint8                          `json:"images_grayscale_single_channel" bson:"images_grayscale_single_channel"`
	VersionOfUsedChocolateLVSDE                     string                             `json:"version_of_used_chocolate_lvsde" bson:"version_of_used_chocolate_lvsde"`
}

type HyperDataAbstractionUnits struct {
	ZeroBasedIndex       int32               `json:"zero_based_index" bson:"zero_based_index"`
	IterationProjections [][]HyperProjection `json:"iteration_projections" bson:"iteration_projections"`
	SingleClassNumber    int32               `json:"single_class_number" bson:"single_class_number"`
	ExtraClassNumbers    []int32             `json:"extra_class_numbers" bson:"extra_class_numbers"`
	ShortTextInfo        string              `json:"short_text_info" bson:"short_text_info"`
	LongTextInfo         string              `json:"long_text_info" bson:"long_text_info"`
	BinaryInfo           [][]uint8           `json:"binary_info" bson:"binary_info"`
	BinaryInfoTypes      []string            `json:"binary_info_types" bson:"binary_info_types"`
}

type HyperProjection struct {
	X               float64   `json:"x" bson:"x"`
	Y               float64   `json:"y" bson:"y"`
	ExtraDimensions []float64 `json:"extra_dimensions" bson:"extra_dimensions"`
	Layer           int32     `json:"layer" bson:"layer"`
}

type EmbeddedData struct {
	FileFormat                             string                      `json:"file_format" bson:"file_format"`
	FileStructureVersion                   []int32                     `json:"file_structure_version" bson:"file_structure_version"`
	DataInstances                          []HyperDataAbstractionUnits `json:"data_instances" bson:"data_instances"`
	IsRedGray                              bool                        `json:"is_red_gray" bson:"is_red_gray"`
	IsStrictRedGray                        bool                        `json:"is_strict_red_gray" bson:"is_strict_red_gray"`
	RedLayerNumber                         *int32                      `json:"red_layer_number" bson:"red_layer_number"`
	GrayLayerNumber                        *int32                      `json:"gray_layer_number" bson:"gray_layer_number"`
	LayerNames                             []string                    `json:"layer_names" bson:"layer_names"`
	EmbeddingMethodName                    string                      `json:"embedding_method_name" bson:"embedding_method_name"`
	DataSetName                            string                      `json:"data_set_name" bson:"data_set_name"`
	EmbeddingMethodParameters              string                      `json:"embedding_method_parameters" bson:"embedding_method_parameters"`
	HasImages                              bool                        `json:"has_images" bson:"has_images"`
	HasShortTextInfo                       bool                        `json:"has_short_text_info" bson:"has_short_text_info"`
	HasLongTextInfo                        bool                        `json:"has_long_text_info" bson:"has_long_text_info"`
	HasMultipleClassNumbersPerDataInstance bool                        `json:"has_multiple_class_numbers_per_data_instance" bson:"has_multiple_class_numbers_per_data_instance"`
	SingleClassLabels                      []string                    `json:"single_class_labels" bson:"single_class_labels"`
	ExtraClassesLabels                     [][]string                  `json:"extra_classes_labels" bson:"extra_classes_labels"`
}

func (dataAbstractionUnit *DataAbstractionUnit) SetDefaultValues() {
	dataAbstractionUnit.ClassLabelNumber = -1

	dataAbstractionUnit.VisualSpaceCoordinates = make([][2]float64, 1)
	dataAbstractionUnit.VisualSpaceCoordinates[0][0] = 0
	dataAbstractionUnit.VisualSpaceCoordinates[0][1] = 0

	dataAbstractionUnit.TemporaryVisualSpaceCoordinates = make([][2]float64, 1)
	dataAbstractionUnit.TemporaryVisualSpaceCoordinates[0][0] = 0
	dataAbstractionUnit.TemporaryVisualSpaceCoordinates[0][1] = 0

	dataAbstractionUnit.AreAllVisualSpaceProjectionsIneffective = false
	dataAbstractionUnit.AreAllVisualSpaceProjectionsFrozen = false

	dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer = true

	dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis = make([][36]float64, 1)
	dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis = make([][36]float64, 1)

	for i := 0; i < 36; i++ {
		dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis[0][i] = 0
		dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis[0][i] = 0
	}

	dataAbstractionUnit.ImageRGB = nil
	dataAbstractionUnit.ImageGrayscale = nil
	dataAbstractionUnit.DataAbstractionUnitNumber = -1
	dataAbstractionUnit.OriginalSpaceCoordinates = nil
	dataAbstractionUnit.ThirtyDimensionalSpaceCoordinates = nil

	dataAbstractionUnit.Mass = make([]float64, 1)
	dataAbstractionUnit.Mass[0] = 1
	dataAbstractionUnit.HasVertexSplitFailed = false
}

func (dataAbstractionSet *DataAbstractionSet) SetDefaultValues(numberOfDataAbstractionUnits int32) {
	dataAbstractionSet.DataAbstractionUnits = make([]DataAbstractionUnit, numberOfDataAbstractionUnits)
	dataAbstractionSet.DistancesBeforeTransformation = nil
	dataAbstractionSet.DistancesAfterTransformation = nil
	dataAbstractionSet.DistancesBeforeThirtyDimensionalUMAP = nil
}

func (dataAbstractionSet *DataAbstractionSet) ComputeDistancesBeforeTransformationEuclidean() {
	numberOfDataAbstractionUnits := len(dataAbstractionSet.DataAbstractionUnits)
	dataAbstractionSet.DistancesBeforeTransformation = make([][]float64, numberOfDataAbstractionUnits)

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionSet.DistancesBeforeTransformation[i] = make([]float64, numberOfDataAbstractionUnits)
	}

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		for j := 0; j < numberOfDataAbstractionUnits; j++ {
			if i == j {
				dataAbstractionSet.DistancesBeforeTransformation[i][j] = 0
				continue
			}

			var distance float64 = 0
			for k := 0; k < len(dataAbstractionSet.DataAbstractionUnits[i].OriginalSpaceCoordinates); k++ {
				distance += math.Pow(dataAbstractionSet.DataAbstractionUnits[i].OriginalSpaceCoordinates[k]-dataAbstractionSet.DataAbstractionUnits[j].OriginalSpaceCoordinates[k], 2)
			}
			distance = math.Sqrt(distance)

			dataAbstractionSet.DistancesBeforeTransformation[i][j] = distance
		}
	}
}

func (dataAbstractionSet *DataAbstractionSet) ComputeDistancesBeforeTransformationCosine() {
	numberOfDataAbstractionUnits := len(dataAbstractionSet.DataAbstractionUnits)
	dataAbstractionSet.DistancesBeforeTransformation = make([][]float64, numberOfDataAbstractionUnits)

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionSet.DistancesBeforeTransformation[i] = make([]float64, numberOfDataAbstractionUnits)
	}

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		for j := 0; j < numberOfDataAbstractionUnits; j++ {
			if i == j {
				dataAbstractionSet.DistancesBeforeTransformation[i][j] = 0
				continue
			}

			var t1, t2, t3 float64 = 0, 0, 0
			for k := 0; k < len(dataAbstractionSet.DataAbstractionUnits[i].OriginalSpaceCoordinates); k++ {
				t1 += dataAbstractionSet.DataAbstractionUnits[i].OriginalSpaceCoordinates[k] * dataAbstractionSet.DataAbstractionUnits[j].OriginalSpaceCoordinates[k]
				t2 += math.Pow(dataAbstractionSet.DataAbstractionUnits[i].OriginalSpaceCoordinates[k], 2)
				t3 += math.Pow(dataAbstractionSet.DataAbstractionUnits[j].OriginalSpaceCoordinates[k], 2)
			}

			var distance float64 = 1.0 - t1/(math.Sqrt(t2)*math.Sqrt(t3))
			if math.Abs(t1) < 1e-6 {
				distance = 1.0
			}

			dataAbstractionSet.DistancesBeforeTransformation[i][j] = distance
		}
	}
}

func (dataAbstractionSet *DataAbstractionSet) ComputeDistancesBeforeTransformationFromThirtyDimensionalSpaceEuclidean() {
	numberOfDataAbstractionUnits := len(dataAbstractionSet.DataAbstractionUnits)
	dataAbstractionSet.DistancesBeforeTransformation = make([][]float64, numberOfDataAbstractionUnits)

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionSet.DistancesBeforeTransformation[i] = make([]float64, numberOfDataAbstractionUnits)
	}

	for i := 0; i < numberOfDataAbstractionUnits; i++ {
		for j := 0; j < numberOfDataAbstractionUnits; j++ {
			if i == j {
				dataAbstractionSet.DistancesBeforeTransformation[i][j] = 0
				continue
			}

			var distance float64 = 0
			for k := 0; k < len(dataAbstractionSet.DataAbstractionUnits[i].ThirtyDimensionalSpaceCoordinates); k++ {
				distance += math.Pow(dataAbstractionSet.DataAbstractionUnits[i].ThirtyDimensionalSpaceCoordinates[k]-dataAbstractionSet.DataAbstractionUnits[j].ThirtyDimensionalSpaceCoordinates[k], 2)
			}
			distance = math.Sqrt(distance)

			dataAbstractionSet.DistancesBeforeTransformation[i][j] = distance
		}
	}
}

func (dataAbstractionSet *DataAbstractionSet) ComputeDistancesAfterTransformation() {
	numberOfDataAbstractionUnits := int32(len(dataAbstractionSet.DataAbstractionUnits))

	var i, j int32

	dataAbstractionSet.DistancesAfterTransformation = make([][]float64, numberOfDataAbstractionUnits)

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionSet.DistancesAfterTransformation[i] = make([]float64, numberOfDataAbstractionUnits)
	}

	m := make([]float64, numberOfDataAbstractionUnits)

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		originalSpaceDistanceCompare := func(a, b interface{}) int {
			indexA := a.(*DataAbstractionUnit).DataAbstractionUnitNumber
			indexB := b.(*DataAbstractionUnit).DataAbstractionUnitNumber
			if dataAbstractionSet.DistancesBeforeTransformation[i][indexA] < dataAbstractionSet.DistancesBeforeTransformation[i][indexB] {
				return -1
			} else if dataAbstractionSet.DistancesBeforeTransformation[i][indexA] == dataAbstractionSet.DistancesBeforeTransformation[i][indexB] {
				return 0
			} else {
				return 1
			}
		}

		queue := priorityqueue.NewWith(originalSpaceDistanceCompare)

		for j = 0; j < numberOfDataAbstractionUnits; j++ {
			if i == j {
				continue
			}
			dataAbstractionUnit2 := &dataAbstractionSet.DataAbstractionUnits[j]
			queue.Enqueue(dataAbstractionUnit2)
		}

		for j = 0; j < 20; j++ {
			dataAbstractionUnit2, _ := queue.Dequeue()
			if dataAbstractionUnit2 == nil {
				panic("Not finished successfully.")
			}
			neighbourIndex := dataAbstractionUnit2.(*DataAbstractionUnit).DataAbstractionUnitNumber
			if j == 19 {
				m[i] = math.Tan(1.0) / dataAbstractionSet.DistancesBeforeTransformation[i][neighbourIndex]
			}
		}
	}
	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		for j = 0; j < numberOfDataAbstractionUnits; j++ {
			distance := dataAbstractionSet.DistancesBeforeTransformation[i][j]
			dataAbstractionSet.DistancesAfterTransformation[i][j] = (math.Atan(m[i]*distance) + math.Atan(m[j]*distance)) / 2.0
		}
	}
}

func (dataAbstractionUnit *DataAbstractionUnit) Copy() *DataAbstractionUnit {
	dataAbstractionUnitCopy := new(DataAbstractionUnit)

	dataAbstractionUnitCopy.OriginalSpaceCoordinates = make([]float64, len(dataAbstractionUnit.OriginalSpaceCoordinates))
	copy(dataAbstractionUnitCopy.OriginalSpaceCoordinates, dataAbstractionUnit.OriginalSpaceCoordinates)

	dataAbstractionUnitCopy.ClassLabelNumber = dataAbstractionUnit.ClassLabelNumber

	dataAbstractionUnitCopy.VisualSpaceCoordinates = make([][2]float64, len(dataAbstractionUnit.VisualSpaceCoordinates))
	copy(dataAbstractionUnitCopy.VisualSpaceCoordinates, dataAbstractionUnit.VisualSpaceCoordinates)

	dataAbstractionUnitCopy.VisualSpacePositiveReplicationPressuresPerAxis = make([][36]float64, len(dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis))
	copy(dataAbstractionUnitCopy.VisualSpacePositiveReplicationPressuresPerAxis, dataAbstractionUnit.VisualSpacePositiveReplicationPressuresPerAxis)

	dataAbstractionUnitCopy.VisualSpaceNegativeReplicationPressuresPerAxis = make([][36]float64, len(dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis))
	copy(dataAbstractionUnitCopy.VisualSpaceNegativeReplicationPressuresPerAxis, dataAbstractionUnit.VisualSpaceNegativeReplicationPressuresPerAxis)

	dataAbstractionUnitCopy.TemporaryVisualSpaceCoordinates = make([][2]float64, len(dataAbstractionUnit.TemporaryVisualSpaceCoordinates))
	copy(dataAbstractionUnitCopy.TemporaryVisualSpaceCoordinates, dataAbstractionUnit.TemporaryVisualSpaceCoordinates)

	dataAbstractionUnitCopy.ImageRGB = dataAbstractionUnit.ImageRGB

	dataAbstractionUnitCopy.ImageGrayscale = dataAbstractionUnit.ImageGrayscale

	dataAbstractionUnitCopy.DataAbstractionUnitNumber = dataAbstractionUnit.DataAbstractionUnitNumber

	dataAbstractionUnitCopy.AreAllVisualSpaceProjectionsIneffective = dataAbstractionUnit.AreAllVisualSpaceProjectionsIneffective

	dataAbstractionUnitCopy.AreAllVisualSpaceProjectionsInRedLayer = dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer

	dataAbstractionUnitCopy.Mass = make([]float64, len(dataAbstractionUnit.Mass))
	copy(dataAbstractionUnitCopy.Mass, dataAbstractionUnit.Mass)

	dataAbstractionUnitCopy.NeighbourIndices = dataAbstractionUnit.NeighbourIndices

	dataAbstractionUnitCopy.HasVertexSplitFailed = dataAbstractionUnit.HasVertexSplitFailed

	dataAbstractionUnitCopy.ComparisonVisualSpaceCoordinates = dataAbstractionUnit.ComparisonVisualSpaceCoordinates

	return dataAbstractionUnitCopy
}

func (dataAbstractionUnit *DataAbstractionUnit) ToDataAbstractionUnitVisibility(iteration int32, method string) *DataAbstractionUnitVisibility {
	dataAbstractionUnitVisibility := new(DataAbstractionUnitVisibility)

	dataAbstractionUnitVisibility.ClassLabelNumber = dataAbstractionUnit.ClassLabelNumber

	dataAbstractionUnitVisibility.DataAbstractionUnitNumber = dataAbstractionUnit.DataAbstractionUnitNumber

	if method == "LVSDE" {
		dataAbstractionUnitVisibility.VisualSpaceCoordinates = make([][2]float64, len(dataAbstractionUnit.VisualSpaceCoordinates))
		copy(dataAbstractionUnitVisibility.VisualSpaceCoordinates, dataAbstractionUnit.VisualSpaceCoordinates)

		if dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer {
			dataAbstractionUnitVisibility.Layer = "red"
		} else {
			dataAbstractionUnitVisibility.Layer = "gray"
		}
	} else {
		dataAbstractionUnitVisibility.VisualSpaceCoordinates = make([][2]float64, 1)
		dataAbstractionUnitVisibility.Layer = "NA"
		if method == "UMAP" {
			dataAbstractionUnitVisibility.VisualSpaceCoordinates[0][0] = dataAbstractionUnit.ComparisonVisualSpaceCoordinates[0][0]
			dataAbstractionUnitVisibility.VisualSpaceCoordinates[0][1] = dataAbstractionUnit.ComparisonVisualSpaceCoordinates[0][1]
		} else {
			dataAbstractionUnitVisibility.VisualSpaceCoordinates[0][0] = dataAbstractionUnit.ComparisonVisualSpaceCoordinates[1][0]
			dataAbstractionUnitVisibility.VisualSpaceCoordinates[0][1] = dataAbstractionUnit.ComparisonVisualSpaceCoordinates[1][1]
		}
	}

	dataAbstractionUnitVisibility.Iteration = iteration

	return dataAbstractionUnitVisibility
}

func (dataAbstractionUnit *DataAbstractionUnit) CreateImageGrayscale() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, int(dataAbstractionUnit.ImageWidth), int(dataAbstractionUnit.ImageHeight)))

	for i := 0; i < int(dataAbstractionUnit.ImageHeight); i++ {
		for j := 0; j < int(dataAbstractionUnit.ImageWidth); j++ {
			grayscale := dataAbstractionUnit.ImageGrayscale[i*int(dataAbstractionUnit.ImageWidth)+j]
			img.Set(j, i, color.RGBA{R: grayscale, G: grayscale, B: grayscale, A: 255})
		}
	}

	return img
}

func (dataAbstractionUnit *DataAbstractionUnit) CreateImageGrayscaleGray() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, int(dataAbstractionUnit.ImageWidth), int(dataAbstractionUnit.ImageHeight)))

	for i := 0; i < int(dataAbstractionUnit.ImageHeight); i++ {
		for j := 0; j < int(dataAbstractionUnit.ImageWidth); j++ {
			grayscale := dataAbstractionUnit.ImageGrayscale[i*int(dataAbstractionUnit.ImageWidth)+j]
			grayscale2 := uint8(127 + int32(grayscale)*128/255)
			img.Set(j, i, color.RGBA{R: grayscale2, G: grayscale2, B: grayscale2, A: 255})
		}
	}

	return img
}

func (dataAbstractionUnit *DataAbstractionUnit) CreateImageGrayscaleRed() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, int(dataAbstractionUnit.ImageWidth), int(dataAbstractionUnit.ImageHeight)))

	for i := 0; i < int(dataAbstractionUnit.ImageHeight); i++ {
		for j := 0; j < int(dataAbstractionUnit.ImageWidth); j++ {
			grayscale := dataAbstractionUnit.ImageGrayscale[i*int(dataAbstractionUnit.ImageWidth)+j]
			img.Set(j, i, color.RGBA{R: grayscale, G: 0, B: 0, A: 255})
		}
	}

	return img
}

func (dataAbstractionUnit *DataAbstractionUnit) CreateImageRGB() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, int(dataAbstractionUnit.ImageWidth), int(dataAbstractionUnit.ImageHeight)))

	for i := 0; i < int(dataAbstractionUnit.ImageHeight); i++ {
		for j := 0; j < int(dataAbstractionUnit.ImageWidth); j++ {
			red := dataAbstractionUnit.ImageRGB[i*int(dataAbstractionUnit.ImageWidth)+j]
			green := dataAbstractionUnit.ImageRGB[int(dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight)+i*int(dataAbstractionUnit.ImageWidth)+j]
			blue := dataAbstractionUnit.ImageRGB[int(dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight*2)+i*int(dataAbstractionUnit.ImageWidth)+j]
			img.Set(j, i, color.RGBA{R: red, G: green, B: blue, A: 255})
		}
	}

	return img
}

func (embeddingDetails *EmbeddingDetails) ToEmbeddedData() *EmbeddedData {
	embeddedData := new(EmbeddedData)

	embeddedData.FileFormat = "Versatile Cartesian Embedded Data File Format (VCED)"
	embeddedData.FileStructureVersion = []int32{1, 0, 0}
	embeddedData.DataInstances = make([]HyperDataAbstractionUnits, len(embeddingDetails.EmbeddingIterations[0]))
	for i := 0; i < len(embeddingDetails.EmbeddingIterations[0]); i++ {
		embeddedData.DataInstances[i].IterationProjections = make([][]HyperProjection, len(embeddingDetails.EmbeddingIterations))
		embeddedData.DataInstances[i].ZeroBasedIndex = embeddingDetails.EmbeddingIterations[0][i].DataAbstractionUnitNumber
		embeddedData.DataInstances[i].SingleClassNumber = embeddingDetails.EmbeddingIterations[0][i].ClassLabelNumber
		embeddedData.DataInstances[i].ExtraClassNumbers = []int32{}
		embeddedData.DataInstances[i].ShortTextInfo = ""
		embeddedData.DataInstances[i].LongTextInfo = ""
		if embeddingDetails.ImagesRedGreenBlueChannels != nil && embeddingDetails.ImagesGrayscaleSingleChannel != nil {
			embeddedData.DataInstances[i].BinaryInfo = [][]uint8{embeddingDetails.ImagesRedGreenBlueChannels[i], embeddingDetails.ImagesGrayscaleSingleChannel[i]}
			embeddedData.DataInstances[i].BinaryInfoTypes = []string{"ImageRGB", "ImageGrayscale"}
		} else if embeddingDetails.ImagesRedGreenBlueChannels != nil {
			embeddedData.DataInstances[i].BinaryInfo = [][]uint8{embeddingDetails.ImagesRedGreenBlueChannels[i]}
			embeddedData.DataInstances[i].BinaryInfoTypes = []string{"ImageRGB"}
		} else if embeddingDetails.ImagesGrayscaleSingleChannel != nil {
			embeddedData.DataInstances[i].BinaryInfo = [][]uint8{embeddingDetails.ImagesGrayscaleSingleChannel[i]}
			embeddedData.DataInstances[i].BinaryInfoTypes = []string{"ImageGrayscale"}
		} else {
			embeddedData.DataInstances[i].BinaryInfo = [][]uint8{}
			embeddedData.DataInstances[i].BinaryInfoTypes = []string{}
		}

	}
	for i := 0; i < len(embeddingDetails.EmbeddingIterations); i++ {
		for j := 0; j < len(embeddingDetails.EmbeddingIterations[i]); j++ {
			embeddedData.DataInstances[j].IterationProjections[i] = make([]HyperProjection, len(embeddingDetails.EmbeddingIterations[i][j].VisualSpaceCoordinates))
			for k := 0; k < len(embeddingDetails.EmbeddingIterations[i][j].VisualSpaceCoordinates); k++ {
				embeddedData.DataInstances[j].IterationProjections[i][k].X = embeddingDetails.EmbeddingIterations[i][j].VisualSpaceCoordinates[k][0]
				embeddedData.DataInstances[j].IterationProjections[i][k].Y = embeddingDetails.EmbeddingIterations[i][j].VisualSpaceCoordinates[k][1]
				embeddedData.DataInstances[j].IterationProjections[i][k].Layer = 0
				if embeddingDetails.EmbeddingIterations[i][j].Layer == "gray" {
					embeddedData.DataInstances[j].IterationProjections[i][k].Layer = 1
				}
				embeddedData.DataInstances[j].IterationProjections[i][k].ExtraDimensions = []float64{}
			}
		}
	}
	embeddedData.IsRedGray = true
	embeddedData.IsStrictRedGray = true
	embeddedData.RedLayerNumber = new(int32)
	*(embeddedData.RedLayerNumber) = 0
	embeddedData.GrayLayerNumber = new(int32)
	*(embeddedData.GrayLayerNumber) = 1
	embeddedData.LayerNames = []string{"Red", "Gray"}
	embeddedData.EmbeddingMethodName = "LVSDE"
	embeddedData.DataSetName = "Not specified"
	embeddedData.EmbeddingMethodParameters = strings.Join([]string{embeddingDetails.VisualDensityAdjustmentParameter, embeddingDetails.NumberOfNeighboursForBuildingNeighbourhoodGraph, embeddingDetails.PreliminaryToThirtyDimensionsUMAP}, ",")
	embeddedData.HasImages = embeddingDetails.ImagesGrayscaleSingleChannel != nil || embeddingDetails.ImagesRedGreenBlueChannels != nil
	embeddedData.HasShortTextInfo = false
	embeddedData.HasLongTextInfo = false
	embeddedData.HasMultipleClassNumbersPerDataInstance = false
	embeddedData.SingleClassLabels = embeddingDetails.ClassLabels
	embeddedData.ExtraClassesLabels = [][]string{}
	return embeddedData
}
