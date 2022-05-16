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

package EmbeddingSpecification

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataEmbedding"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/FileReadingOrWriting"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/PythonInterop"
	"gopkg.in/mgo.v2/bson"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type EmbeddingSpecification struct {
	InputFilePath                                   string   `json:"input_file_path"`
	IsInputFileDistances                            string   `json:"is_input_file_distances"`
	OutputDirectory                                 string   `json:"output_directory"`
	ClassLabels                                     []string `json:"class_labels"`
	ColoursList                                     []string `json:"colours_list"`
	ImagesFileRedGreenBlueChannels                  string   `json:"images_file_red_green_blue_channels"`
	ImagesFileGrayscaleSingleChannel                string   `json:"images_file_grayscale_single_channel"`
	ImagesFileImageWidth                            string   `json:"images_file_image_width"`
	ImagesFileHasClassLabelNumbers                  string   `json:"images_file_has_class_label_numbers"`
	RandomSeed                                      string   `json:"random_seed"`
	RandomState                                     string   `json:"random_state"`
	PreliminaryToThirtyDimensionsUMAP               string   `json:"preliminary_to_thirty_dimensions_umap"`
	NumberOfInitialDataAbstractionUnits             string   `json:"number_of_initial_data_abstraction_units"`
	NumberOfSecondaryDataAbstractionUnits           string   `json:"number_of_secondary_data_abstraction_units"`
	VisualDensityAdjustmentParameter                string   `json:"visual_density_adjustment_parameter"`
	NumberOfNeighboursForBuildingNeighbourhoodGraph string   `json:"number_of_neighbours_for_building_neighbourhood_graph"`
	EvaluationNeighbourhoodSizes                    []string `json:"evaluation_neighbourhood_sizes"`
	CompareWithOtherMethods                         string   `json:"compare_with_other_methods"`
	UseCosineDistanceForInputMultiDimensionalData   string   `json:"use_cosine_distance_for_input_multi_dimensional_data"`
}

type EmbeddingSpecifications struct {
	EmbeddingSpecifications []EmbeddingSpecification `json:"embedding_specifications"`
}

func ReadEmbeddingSpecification(embeddingSpecificationFilePath string) EmbeddingSpecifications {
	_, err := os.Stat(embeddingSpecificationFilePath)
	if os.IsNotExist(err) {
		panic("Not finished successfully. Embedding specifications file does not exist.")
	}

	var embeddingSpecifications EmbeddingSpecifications

	var embeddingSpecificationFile *os.File

	embeddingSpecificationFile, err = os.Open(embeddingSpecificationFilePath)

	if err != nil {
		panic("Not finished successfully. Could not read the embedding specifications file")
	}

	defer embeddingSpecificationFile.Close()

	embeddingSpecificationFileBytes, _ := ioutil.ReadAll(embeddingSpecificationFile)

	err = json.Unmarshal(embeddingSpecificationFileBytes, &embeddingSpecifications)

	if err != nil {
		panic("Not finished successfully. Could not parse the embedding specifications file.")
	}

	for i := 0; i < len(embeddingSpecifications.EmbeddingSpecifications); i++ {
		embeddingSpecification := &embeddingSpecifications.EmbeddingSpecifications[i]

		embeddingSpecification.InputFilePath = filepath.Join(filepath.Dir(embeddingSpecificationFilePath), embeddingSpecification.InputFilePath)
		embeddingSpecification.OutputDirectory = filepath.Join(filepath.Dir(embeddingSpecificationFilePath), embeddingSpecification.OutputDirectory)

		if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" {
			embeddingSpecification.ImagesFileGrayscaleSingleChannel = filepath.Join(filepath.Dir(embeddingSpecificationFilePath), embeddingSpecification.ImagesFileGrayscaleSingleChannel)
		}

		if embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
			embeddingSpecification.ImagesFileRedGreenBlueChannels = filepath.Join(filepath.Dir(embeddingSpecificationFilePath), embeddingSpecification.ImagesFileRedGreenBlueChannels)
		}
	}

	return embeddingSpecifications
}

func RunEmbeddingSpecifications(embeddingSpecifications []EmbeddingSpecification) {
	for i := 0; i < len(embeddingSpecifications); i++ {
		runtime.GC()
		embeddingSpecification := embeddingSpecifications[i]
		var dataAbstractionSet DataAbstraction.DataAbstractionSet

		_, err := os.Stat(embeddingSpecification.OutputDirectory)
		if os.IsNotExist(err) {
			os.MkdirAll(embeddingSpecification.OutputDirectory, 600)
		} else {
			panic("Not finished successfully. Output directory cannot be created because it exists.")
		}

		coloursList := []string{"#8AB9F1", "#6F4E37", "#00FF00", "#8B008B", "#00356B", "#E1A95F", "#4F7942", "#FF66CC", "#F4C430", "#8806CE"}
		if embeddingSpecification.ColoursList != nil {
			coloursList = embeddingSpecification.ColoursList
		}

		classLabels := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
		if embeddingSpecification.ClassLabels != nil {
			classLabels = embeddingSpecification.ClassLabels
		}

		if embeddingSpecification.ColoursList == nil && embeddingSpecification.ClassLabels != nil {
			if len(classLabels) > len(coloursList) {
				panic("Not finished successfully. Inconsistent embedding specifications file.")
			}
			coloursList = coloursList[:len(classLabels)]
		}

		if len(coloursList) != len(classLabels) {
			panic("Not finished successfully. Inconsistent embedding specifications file.")
		}

		var isInputFileDistances bool = false
		if embeddingSpecification.IsInputFileDistances == "true" {
			isInputFileDistances = true
			numberOfInitialDataAbstractionUnits, err := strconv.ParseInt(embeddingSpecification.NumberOfInitialDataAbstractionUnits, 10, 32)
			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
			fmt.Println("Reading input file...")
			dataAbstractionSet = FileReadingOrWriting.ReadDataAbstractionSetFromDistancesFile(embeddingSpecification.InputFilePath, int32(numberOfInitialDataAbstractionUnits), int32(len(coloursList)-1))
			fmt.Println("Reading input file finished.")
		} else if embeddingSpecification.IsInputFileDistances == "false" {
			isInputFileDistances = false
			numberOfInitialDataAbstractionUnits, err := strconv.ParseInt(embeddingSpecification.NumberOfInitialDataAbstractionUnits, 10, 32)
			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
			fmt.Println("Reading input file...")
			dataAbstractionSet = FileReadingOrWriting.ReadDataAbstractionSetFromMultiDimensionalDataFile(embeddingSpecification.InputFilePath, int32(numberOfInitialDataAbstractionUnits), int32(len(coloursList)-1))
			fmt.Println("Reading input file finished.")
		} else {
			panic("Not finished successfully. Could not parse the embedding specifications file.")
		}

		var dataEmbeddingTechniqueLVSDE DataEmbedding.DataEmbeddingTechniqueLVSDE
		if embeddingSpecification.VisualDensityAdjustmentParameter == "" {
			dataEmbeddingTechniqueLVSDE.VisualDensityAdjustmentParameter = 0.9
		} else {
			var err error
			dataEmbeddingTechniqueLVSDE.VisualDensityAdjustmentParameter, err = strconv.ParseFloat(embeddingSpecification.VisualDensityAdjustmentParameter, 64)
			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
		}

		if embeddingSpecification.NumberOfNeighboursForBuildingNeighbourhoodGraph == "" {
			dataEmbeddingTechniqueLVSDE.NumberOfNeighboursForBuildingNeighbourhoodGraph = int32(len(dataAbstractionSet.DataAbstractionUnits) / 3)
		} else {
			numberOfNeighboursForBuildingNeighbourhoodGraph, err := strconv.ParseInt(embeddingSpecification.NumberOfNeighboursForBuildingNeighbourhoodGraph, 10, 32)
			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
			dataEmbeddingTechniqueLVSDE.NumberOfNeighboursForBuildingNeighbourhoodGraph = int32(numberOfNeighboursForBuildingNeighbourhoodGraph)
		}

		var imageWidth int64 = -1
		if embeddingSpecification.ImagesFileImageWidth != "" {
			var err error
			imageWidth, err = strconv.ParseInt(embeddingSpecification.ImagesFileImageWidth, 10, 32)
			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
		}

		if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" && embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
			panic("Not finished successfully. Inconsistent embedding specifications file.")
		}

		if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" {
			if embeddingSpecification.ImagesFileHasClassLabelNumbers == "true" {
				FileReadingOrWriting.ReadImagesFileGrayscaleSingleChannel(embeddingSpecification.ImagesFileGrayscaleSingleChannel, &dataAbstractionSet, int32(imageWidth), true)
			} else if embeddingSpecification.ImagesFileHasClassLabelNumbers == "false" || embeddingSpecification.ImagesFileHasClassLabelNumbers == "" {
				FileReadingOrWriting.ReadImagesFileGrayscaleSingleChannel(embeddingSpecification.ImagesFileGrayscaleSingleChannel, &dataAbstractionSet, int32(imageWidth), false)
			} else {
				panic("Not finished successfully.")
			}
		}

		if embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
			if embeddingSpecification.ImagesFileHasClassLabelNumbers == "true" {
				FileReadingOrWriting.ReadImagesFileRedGreenBlueChannels(embeddingSpecification.ImagesFileRedGreenBlueChannels, &dataAbstractionSet, int32(imageWidth), true)
			} else if embeddingSpecification.ImagesFileHasClassLabelNumbers == "false" || embeddingSpecification.ImagesFileHasClassLabelNumbers == "" {
				FileReadingOrWriting.ReadImagesFileRedGreenBlueChannels(embeddingSpecification.ImagesFileRedGreenBlueChannels, &dataAbstractionSet, int32(imageWidth), false)
			} else {
				panic("Not finished successfully.")
			}
		}

		var useCosineDistance bool = false
		if embeddingSpecification.UseCosineDistanceForInputMultiDimensionalData == "" || embeddingSpecification.UseCosineDistanceForInputMultiDimensionalData == "false" {
			useCosineDistance = false
		} else if embeddingSpecification.UseCosineDistanceForInputMultiDimensionalData == "true" {
			useCosineDistance = true
		} else {
			panic("Not finished successfully. Could not parse the embedding specifications file.")
		}

		if embeddingSpecification.RandomState != "" && embeddingSpecification.RandomSeed != "" {
			panic("Not finished successfully. Inconsistent embedding specifications file.")
		}

		var randomState int64 = 5
		if embeddingSpecification.RandomState != "" {
			var err error
			randomState, err = strconv.ParseInt(embeddingSpecification.RandomState, 10, 32)

			if err != nil || randomState < 0 {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
		}

		compareWithOtherMethods := false
		if embeddingSpecification.CompareWithOtherMethods == "true" {
			compareWithOtherMethods = true
		} else if embeddingSpecification.CompareWithOtherMethods != "" && embeddingSpecification.CompareWithOtherMethods != "false" {
			panic("Not finished successfully. Could not parse the embedding specifications file.")
		}

		preliminaryToThirtyDimensionsUMAP := false
		if embeddingSpecification.PreliminaryToThirtyDimensionsUMAP == "true" || embeddingSpecification.PreliminaryToThirtyDimensionsUMAP == "" {
			preliminaryToThirtyDimensionsUMAP = true
		} else if embeddingSpecification.PreliminaryToThirtyDimensionsUMAP != "false" {
			panic("Not finished successfully. Could not parse the embedding specifications file.")
		}

		comparisonPythonCode := `
		print('Performing UMAP to 2 dimensions for comparison...')
		twoDimUMAP=umap.UMAP(n_components=2, random_state=%d%s).fit_transform(input)
		for i in range(0,numberOfDataAbstractionUnitsOutput):
			for j in range(0,2):
				output.append(float(twoDimUMAP[i,j]))
		print('Two dimensional UMAP for comparison finished.')
		print('Performing t-SNE to 2 dimensions for comparison...')
		twoDimTSNE=TSNE(n_components=2, init='random', learning_rate='auto', method='exact', random_state=%d%s).fit_transform(input)
		for i in range(0,numberOfDataAbstractionUnitsOutput):
			for j in range(0,2):
				output.append(float(twoDimTSNE[i,j]))
		print('Two dimensional t-SNE for comparison finished.')`

		if isInputFileDistances {
			comparisonPythonCode = fmt.Sprintf(comparisonPythonCode, randomState, ", metric='precomputed'", randomState, ", metric='precomputed'")
		} else if useCosineDistance {
			comparisonPythonCode = fmt.Sprintf(comparisonPythonCode, randomState, ", metric='cosine'", randomState, ", metric='cosine'")
		} else {
			comparisonPythonCode = fmt.Sprintf(comparisonPythonCode, randomState, "", randomState, "")
		}

		thirtyDimensionalUmapPythonCode := `
		print('Performing UMAP to 30 dimensions as a preliminary step...')
		thirtyDim=umap.UMAP(n_components=30, random_state=%d%s).fit_transform(input)
		print('Thirty dimensional UMAP as a preliminary step finished.')
		for i in range(0,numberOfDataAbstractionUnitsOutput):
			for j in range(0,30):
				output.append(float(thirtyDim[i,j]))`

		if isInputFileDistances {
			thirtyDimensionalUmapPythonCode = fmt.Sprintf(thirtyDimensionalUmapPythonCode, randomState, ", metric='precomputed'")
		} else if useCosineDistance {
			thirtyDimensionalUmapPythonCode = fmt.Sprintf(thirtyDimensionalUmapPythonCode, randomState, ", metric='cosine'")
		} else {
			thirtyDimensionalUmapPythonCode = fmt.Sprintf(thirtyDimensionalUmapPythonCode, randomState, "")
		}

		if preliminaryToThirtyDimensionsUMAP || compareWithOtherMethods {

			functionCode := `
def SomeDimensionalityReductions(*x):
	import sys
	try:
		numberOfDataAbstractionUnits=int(x[0])
		numberOfDataAbstractionUnitsOutput=int(x[1])
		numberOfDimensions=int((len(x)-2)/numberOfDataAbstractionUnits)
		input=[]
		for i in range(0,numberOfDataAbstractionUnits):
			input.append([])
			for j in range(0,numberOfDimensions):
				input[i].append(x[2+i*numberOfDimensions+j])
		import numpy as np
		import umap
		from sklearn.manifold import TSNE
		import time
		input=np.array(input)
		output=[]`

			if compareWithOtherMethods {
				functionCode += comparisonPythonCode
			}
			if preliminaryToThirtyDimensionsUMAP {
				functionCode += `
		print('LVSDE embedding started at '+time.strftime('%a %b %d %H:%M:%S %Z %Y',time.localtime()))`
				functionCode += thirtyDimensionalUmapPythonCode
			}
			functionCode += `
	except:
		print(str(sys.exc_info()))
	return tuple(output)
`
			var functionParameters []float64
			if isInputFileDistances {
				functionParameters = make([]float64, len(dataAbstractionSet.DataAbstractionUnits)*len(dataAbstractionSet.DataAbstractionUnits)+2)
			} else {
				functionParameters = make([]float64, len(dataAbstractionSet.DataAbstractionUnits)*len(dataAbstractionSet.DataAbstractionUnits[0].OriginalSpaceCoordinates)+2)
			}

			functionParameters[0] = float64(len(dataAbstractionSet.DataAbstractionUnits))
			numberOfSecondaryDataAbstractionUnits := int64(len(dataAbstractionSet.DataAbstractionUnits))
			if embeddingSpecification.NumberOfSecondaryDataAbstractionUnits != "" {
				var err error
				numberOfSecondaryDataAbstractionUnits, err = strconv.ParseInt(embeddingSpecification.NumberOfSecondaryDataAbstractionUnits, 10, 32)
				if err != nil {
					panic("Not finished successfully. Could not parse the embedding specifications file.")
				}
			}
			functionParameters[1] = float64(numberOfSecondaryDataAbstractionUnits)

			if isInputFileDistances {
				for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
					for k := 0; k < len(dataAbstractionSet.DataAbstractionUnits); k++ {
						functionParameters[j*len(dataAbstractionSet.DataAbstractionUnits)+k+2] = dataAbstractionSet.DistancesBeforeTransformation[j][k]
					}
				}
			} else {
				for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
					dataAbstractionUnit := dataAbstractionSet.DataAbstractionUnits[j]
					for k := 0; k < len(dataAbstractionUnit.OriginalSpaceCoordinates); k++ {
						functionParameters[j*len(dataAbstractionUnit.OriginalSpaceCoordinates)+k+2] = dataAbstractionUnit.OriginalSpaceCoordinates[k]
					}
				}
			}

			functionOutputSize := 30 * int(numberOfSecondaryDataAbstractionUnits)
			if compareWithOtherMethods && preliminaryToThirtyDimensionsUMAP {
				functionOutputSize = 34 * int(numberOfSecondaryDataAbstractionUnits)
			} else if compareWithOtherMethods && !preliminaryToThirtyDimensionsUMAP {
				functionOutputSize = 4 * int(numberOfSecondaryDataAbstractionUnits)
			}

			output := PythonInterop.RunPythonFunction(functionCode, "SomeDimensionalityReductions", functionParameters, functionOutputSize)

			dataAbstractionSet.DataAbstractionUnits = dataAbstractionSet.DataAbstractionUnits[:numberOfSecondaryDataAbstractionUnits]

			var outputsUsed int = 0

			if compareWithOtherMethods {
				for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
					dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[j]
					dataAbstractionUnit.ComparisonVisualSpaceCoordinates = make([][2]float64, 2)
					for k := 0; k < 2; k++ {
						dataAbstractionUnit.ComparisonVisualSpaceCoordinates[0][k] = output[j*2+k+outputsUsed]
					}
				}
				outputsUsed += 2 * int(numberOfSecondaryDataAbstractionUnits)

				for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
					dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[j]
					for k := 0; k < 2; k++ {
						dataAbstractionUnit.ComparisonVisualSpaceCoordinates[1][k] = output[j*2+k+outputsUsed]
					}
				}
				outputsUsed += 2 * int(numberOfSecondaryDataAbstractionUnits)
			}

			if preliminaryToThirtyDimensionsUMAP {
				for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
					dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[j]
					dataAbstractionUnit.ThirtyDimensionalSpaceCoordinates = make([]float64, 30)
					for k := 0; k < 30; k++ {
						dataAbstractionUnit.ThirtyDimensionalSpaceCoordinates[k] = output[j*30+k+outputsUsed]
					}
				}
				outputsUsed += 30 * int(numberOfSecondaryDataAbstractionUnits)
			} else {
				fmt.Println("LVSDE embedding started at", time.Now().Format(time.UnixDate))
			}

			if dataAbstractionSet.DistancesBeforeTransformation != nil {
				dataAbstractionSet.DistancesBeforeThirtyDimensionalUMAP = dataAbstractionSet.DistancesBeforeThirtyDimensionalUMAP
			}
		}

		if preliminaryToThirtyDimensionsUMAP {
			dataAbstractionSet.ComputeDistancesBeforeTransformationFromThirtyDimensionalSpaceEuclidean()

		} else {
			if dataAbstractionSet.DistancesBeforeTransformation == nil {
				if useCosineDistance {
					dataAbstractionSet.ComputeDistancesBeforeTransformationCosine()
				} else {
					dataAbstractionSet.ComputeDistancesBeforeTransformationEuclidean()
				}
			}
		}

		dataEmbeddingTechniqueLVSDE.RandomSeed = 159720256358285954
		if embeddingSpecification.RandomSeed != "" {
			var err error
			dataEmbeddingTechniqueLVSDE.RandomSeed, err = strconv.ParseInt(embeddingSpecification.RandomSeed, 10, 64)

			if err != nil {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}
		} else if embeddingSpecification.RandomState != "" {
			var err error
			randomState, err = strconv.ParseInt(embeddingSpecification.RandomState, 10, 32)

			if err != nil || randomState < 0 {
				panic("Not finished successfully. Could not parse the embedding specifications file.")
			}

			randomSeed := dataEmbeddingTechniqueLVSDE.RandomSeed
			randomSource := rand.NewSource(randomSeed)
			randomGenerator := rand.New(randomSource)
			for i := 0; i < int(randomState); i++ {
				randomSeed = randomGenerator.Int63()
			}
		}

		dataEmbeddingTechniqueLVSDE.EmbedData(dataAbstractionSet)

		fmt.Println("Saving to file...,         time:", time.Now().Format(time.UnixDate))

		embeddingDetails := &dataEmbeddingTechniqueLVSDE.EmbeddingDetails
		embeddingDetails.ImageWidth = dataAbstractionSet.DataAbstractionUnits[0].ImageWidth
		embeddingDetails.NumberOfSecondaryDataAbstractionUnits = embeddingSpecification.NumberOfSecondaryDataAbstractionUnits
		embeddingDetails.RandomSeed = embeddingSpecification.RandomSeed
		embeddingDetails.PreliminaryToThirtyDimensionsUMAP = embeddingSpecification.PreliminaryToThirtyDimensionsUMAP
		if embeddingDetails.PreliminaryToThirtyDimensionsUMAP == "" {
			embeddingDetails.PreliminaryToThirtyDimensionsUMAP = "true"
		}
		embeddingDetails.NumberOfNeighboursForBuildingNeighbourhoodGraph = strconv.Itoa(int(dataEmbeddingTechniqueLVSDE.NumberOfNeighboursForBuildingNeighbourhoodGraph))
		embeddingDetails.NumberOfInitialDataAbstractionUnits = embeddingSpecification.NumberOfInitialDataAbstractionUnits
		embeddingDetails.VisualDensityAdjustmentParameter = embeddingSpecification.VisualDensityAdjustmentParameter
		if embeddingDetails.VisualDensityAdjustmentParameter == "" {
			embeddingDetails.VisualDensityAdjustmentParameter = "0.9"
		}
		embeddingDetails.ClassLabels = embeddingSpecification.ClassLabels
		embeddingDetails.ColoursList = coloursList
		embeddingDetails.RandomState = embeddingSpecification.RandomState
		embeddingDetails.EvaluationNeighbourhoodSizes = embeddingSpecification.EvaluationNeighbourhoodSizes

		if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" {
			embeddingDetails.ImagesGrayscaleSingleChannel = make([][]uint8, len(dataAbstractionSet.DataAbstractionUnits))
			for i := 0; i < len(dataAbstractionSet.DataAbstractionUnits); i++ {
				embeddingDetails.ImagesGrayscaleSingleChannel[i] = dataAbstractionSet.DataAbstractionUnits[i].ImageGrayscale
			}
		}

		if embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
			embeddingDetails.ImagesRedGreenBlueChannels = make([][]uint8, len(dataAbstractionSet.DataAbstractionUnits))
			for i := 0; i < len(dataAbstractionSet.DataAbstractionUnits); i++ {
				embeddingDetails.ImagesRedGreenBlueChannels[i] = dataAbstractionSet.DataAbstractionUnits[i].ImageRGB
			}
		}

		embeddingDetails.VersionOfUsedChocolateLVSDE = "1.0"

		lastIteration := 1829

		FileReadingOrWriting.WriteEmbeddingToFile(embeddingDetails.EmbeddingIterations[lastIteration], filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration_colouring_0.png"), 0, &dataAbstractionSet, coloursList)
		FileReadingOrWriting.WriteEmbeddingToFile(embeddingDetails.EmbeddingIterations[lastIteration], filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration_colouring_1.png"), 1, &dataAbstractionSet, coloursList)
		FileReadingOrWriting.WriteEmbeddingToFile(embeddingDetails.EmbeddingIterations[lastIteration], filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration_colouring_2.png"), 2, &dataAbstractionSet, coloursList)
		if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" || embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
			FileReadingOrWriting.WriteEmbeddingToFile(embeddingDetails.EmbeddingIterations[lastIteration], filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration_colouring_3.png"), 3, &dataAbstractionSet, coloursList)
			FileReadingOrWriting.WriteEmbeddingToFile(embeddingDetails.EmbeddingIterations[lastIteration], filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration_colouring_4.png"), 4, &dataAbstractionSet, coloursList)
		}

		jsonBytes, _ := json.MarshalIndent(embeddingDetails.EmbeddingIterations[lastIteration], "", "\t")
		ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "last_iteration.json"), jsonBytes, 640)

		jsonBytes, _ = json.MarshalIndent(embeddingDetails.EmbeddingIterations, "", "\t")
		var zipFile *os.File
		zipFile, err = os.Create(filepath.Join(embeddingSpecification.OutputDirectory, "iterations.json.zip"))
		if err != nil {
			panic("Not finished successfully.")
		}
		zipWriter := zip.NewWriter(zipFile)
		var writer io.Writer
		writer, err = zipWriter.Create("iterations.json")
		if err != nil {
			panic("Not finished successfully.")
		}
		writer.Write(jsonBytes)
		zipWriter.Close()
		zipFile.Close()

		bsonBytes, _ := bson.Marshal(embeddingDetails)
		zipFile, err = os.Create(filepath.Join(embeddingSpecification.OutputDirectory, "embedding_details.bson.zip"))
		if err != nil {
			panic("Not finished successfully.")
		}
		zipWriter = zip.NewWriter(zipFile)
		writer, err = zipWriter.Create("embedding_details.bson")
		if err != nil {
			panic("Not finished successfully.")
		}
		writer.Write(bsonBytes)
		zipWriter.Close()
		zipFile.Close()

		FileReadingOrWriting.WriteLegendFileHtml(filepath.Join(embeddingSpecification.OutputDirectory, "legend.html"), classLabels, coloursList)
		FileReadingOrWriting.WriteShowFileHtml(embeddingSpecification.OutputDirectory, embeddingDetails.EmbeddingIterations[lastIteration])

		fmt.Println("Embedding and saving to file finished on", time.Now().Format(time.UnixDate))

		if len(embeddingSpecification.EvaluationNeighbourhoodSizes) > 0 || compareWithOtherMethods {
			fmt.Println("Please wait...")
		}

		if len(embeddingSpecification.EvaluationNeighbourhoodSizes) > 0 {
			evaluationCSV := strings.Builder{}

			for _, evaluationNeighbourhoodSize := range embeddingSpecification.EvaluationNeighbourhoodSizes {
				k, err := strconv.Atoi(evaluationNeighbourhoodSize)
				if err != nil {
					panic("Not finished successfully.")
				}
				evaluationCSV.WriteString("Evaluation_type,Evaluation_neighbourhood_size,Evaluation\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(red_and_gray)_(red_and_gray)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"red", "gray"}, []string{"red", "gray"}), 'f', 3, 64) + "%\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(red_and_gray)_(red)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"red", "gray"}, []string{"red"}), 'f', 3, 64) + "%\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(red)_(red)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"red"}, []string{"red"}), 'f', 3, 64) + "%\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(gray)_(gray)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"gray"}, []string{"gray"}), 'f', 3, 64) + "%\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(gray)_(red)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"gray"}, []string{"red"}), 'f', 3, 64) + "%\r\n")
				evaluationCSV.WriteString("KNN_accuracy_(gray)_(red_and_gray)," + evaluationNeighbourhoodSize + "," +
					strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
						embeddingDetails.EmbeddingIterations[lastIteration], k,
						[]string{"gray"}, []string{"red", "gray"}), 'f', 3, 64) + "%\r\n")
			}

			err = ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "evaluation.csv"), []byte(evaluationCSV.String()), 600)
			if err != nil {
				panic("Not finished successfully.")
			}
		}

		if compareWithOtherMethods {
			_, err := os.Stat(filepath.Join(embeddingSpecification.OutputDirectory, "compare"))
			if os.IsNotExist(err) {
				os.MkdirAll(filepath.Join(embeddingSpecification.OutputDirectory, "compare"), 600)
			}

			embedding := make([]*DataAbstraction.DataAbstractionUnitVisibility, len(dataAbstractionSet.DataAbstractionUnits))

			for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
				dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[j]
				dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer = true
				embedding[j] = dataAbstractionUnit.ToDataAbstractionUnitVisibility(1, "UMAP")
			}

			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_colouring_0.png"), 0, &dataAbstractionSet, coloursList)
			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_colouring_1.png"), 1, &dataAbstractionSet, coloursList)
			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_colouring_2.png"), 2, &dataAbstractionSet, coloursList)
			if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" || embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
				FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_iteration_colouring_3.png"), 3, &dataAbstractionSet, coloursList)
				FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_iteration_colouring_4.png"), 4, &dataAbstractionSet, coloursList)
			}

			jsonBytes, _ = json.MarshalIndent(embedding, "", "\t")
			ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "compare", "UMAP_embedding.json"), jsonBytes, 640)

			if len(embeddingSpecification.EvaluationNeighbourhoodSizes) > 0 {
				evaluationCSV := strings.Builder{}

				for _, evaluationNeighbourhoodSize := range embeddingSpecification.EvaluationNeighbourhoodSizes {
					k, err := strconv.Atoi(evaluationNeighbourhoodSize)
					if err != nil {
						panic("Not finished successfully.")
					}
					evaluationCSV.WriteString("Evaluation_type,Evaluation_neighbourhood_size,Evaluation\r\n")
					evaluationCSV.WriteString("KNN_accuracy," + evaluationNeighbourhoodSize + "," +
						strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
							embedding, k,
							[]string{"NA"}, []string{"NA"}), 'f', 3, 64) + "%\r\n")
				}

				err = ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "evaluation_UMAP.csv"), []byte(evaluationCSV.String()), 600)
				if err != nil {
					panic("Not finished successfully.")
				}
			}

			embedding = make([]*DataAbstraction.DataAbstractionUnitVisibility, len(dataAbstractionSet.DataAbstractionUnits))

			for j := 0; j < len(dataAbstractionSet.DataAbstractionUnits); j++ {
				dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[j]
				dataAbstractionUnit.AreAllVisualSpaceProjectionsInRedLayer = true
				embedding[j] = dataAbstractionUnit.ToDataAbstractionUnitVisibility(1, "t-SNE")
			}

			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_colouring_0.png"), 0, &dataAbstractionSet, coloursList)
			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_colouring_1.png"), 1, &dataAbstractionSet, coloursList)
			FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_colouring_2.png"), 2, &dataAbstractionSet, coloursList)
			if embeddingSpecification.ImagesFileGrayscaleSingleChannel != "" || embeddingSpecification.ImagesFileRedGreenBlueChannels != "" {
				FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_iteration_colouring_3.png"), 3, &dataAbstractionSet, coloursList)
				FileReadingOrWriting.WriteEmbeddingToFile(embedding, filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_iteration_colouring_4.png"), 4, &dataAbstractionSet, coloursList)
			}

			jsonBytes, _ = json.MarshalIndent(embedding, "", "\t")
			ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "compare", "t-SNE_embedding.json"), jsonBytes, 640)

			if len(embeddingSpecification.EvaluationNeighbourhoodSizes) > 0 {
				evaluationCSV := strings.Builder{}

				for _, evaluationNeighbourhoodSize := range embeddingSpecification.EvaluationNeighbourhoodSizes {
					k, err := strconv.Atoi(evaluationNeighbourhoodSize)
					if err != nil {
						panic("Not finished successfully.")
					}
					evaluationCSV.WriteString("Evaluation_type,Evaluation_neighbourhood_size,Evaluation\r\n")
					evaluationCSV.WriteString("KNN_accuracy," + evaluationNeighbourhoodSize + "," +
						strconv.FormatFloat(DataEmbedding.EvaluateEmbedding(
							embedding, k,
							[]string{"NA"}, []string{"NA"}), 'f', 3, 64) + "%\r\n")
				}

				err = ioutil.WriteFile(filepath.Join(embeddingSpecification.OutputDirectory, "evaluation_t-SNE.csv"), []byte(evaluationCSV.String()), 600)
				if err != nil {
					panic("Not finished successfully.")
				}
			}
		}

		if len(embeddingSpecification.EvaluationNeighbourhoodSizes) > 0 || compareWithOtherMethods {
			fmt.Println("Finished processing the embedding specification file.")
		}
	}
}
