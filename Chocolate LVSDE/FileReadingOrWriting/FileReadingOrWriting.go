/*
	Named "Chocolate LVSDE", this project (including but not limited to this file) is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.
	LVSDE stands for Layered Vertex Splitting Data Embedding.
	LVSDE dimensionality reduction technique (algorithm) was first named Red Gray Plus (subject to possibly some changes or corrections). Red Gray Plus is described in the arXiv preprint numbered 2101.06224 which should be updated soon (not yet) to use the name LVSDE instead.

	The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

	Copyright notice for this code (this implementation of LVSDE) and this file:
	Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.

	All codes in this project including but not limited to this file are written by Farshad Barahimi.

	The purpose of writing this code is academic.

	LVSDE stands for Layered Vertex Splitting Data Embedding.
	For more information about LVSDE dimensionality reduction technique (algorithm) look at the following arXiv preprint:
	Farshad Barahimi, "Multi-point dimensionality reduction to improve projection layout reliability",  arXiv:2101.06224v3, 2022.
*/

package FileReadingOrWriting

import (
	"bufio"
	"fmt"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/WebUserInterface"
	"io"
	"io/fs"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

import "github.com/fogleman/gg"

var Chmod fs.FileMode = 0700

func ReadDataAbstractionSetFromDistancesFile(filePath string, numberOfInitialDataAbstractionUnits int32, maximumClassLabelNumber int32) DataAbstraction.DataAbstractionSet {
	var dataAbstractionSet DataAbstraction.DataAbstractionSet

	file, err := os.Open(filePath)

	if err != nil {
		panic("File read error")
	}

	reader := bufio.NewReader(file)

	var dataAbstractionUnitNumber int32 = -1
	dataAbstractionSet.SetDefaultValues(int32(numberOfInitialDataAbstractionUnits))
	dataAbstractionSet.DistancesBeforeTransformation = make([][]float64, int32(numberOfInitialDataAbstractionUnits))

	for i := 0; i < int(numberOfInitialDataAbstractionUnits); i++ {
		dataAbstractionSet.DistancesBeforeTransformation[i] = make([]float64, int32(numberOfInitialDataAbstractionUnits))
	}

	for {
		read, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}

		read = strings.TrimSpace(read)

		if len(read) == 0 {
			if err != nil {
				break
			}
			continue
		}

		readNumbers := strings.Split(read, ",")

		dataAbstractionUnitNumber++

		if dataAbstractionUnitNumber == numberOfInitialDataAbstractionUnits {
			break
		}

		if dataAbstractionUnitNumber%5000 == 0 && dataAbstractionUnitNumber != 0 {
			fmt.Println("Current number of data abstraction units read:", dataAbstractionUnitNumber, ", more to read...")
		}

		var classLabelNumber int64
		classLabelNumber, _ = strconv.ParseInt(readNumbers[0], 10, 32)
		if classLabelNumber > int64(maximumClassLabelNumber) {
			panic("Not finished successfully. Not enough colours specified for class label numbers.")
		}

		var dataAbstractionUnit DataAbstraction.DataAbstractionUnit
		dataAbstractionUnit.SetDefaultValues()
		dataAbstractionUnit.ClassLabelNumber = int32(classLabelNumber)

		for i := 1; i <= int(numberOfInitialDataAbstractionUnits); i++ {
			dataAbstractionSet.DistancesBeforeTransformation[dataAbstractionUnitNumber][i-1], _ = strconv.ParseFloat(readNumbers[i], 64)
		}

		dataAbstractionUnit.DataAbstractionUnitNumber = dataAbstractionUnitNumber

		dataAbstractionSet.DataAbstractionUnits[dataAbstractionUnitNumber] = dataAbstractionUnit

		if err != nil {
			break
		}
	}

	return dataAbstractionSet
}

func ReadDataAbstractionSetFromMultiDimensionalDataFile(filePath string, numberOfInitialDataAbstractionUnits int32, maximumClassLabelNumber int32) DataAbstraction.DataAbstractionSet {
	var dataAbstractionSet DataAbstraction.DataAbstractionSet

	file, err := os.Open(filePath)

	if err != nil {
		panic("File read error")
	}

	reader := bufio.NewReader(file)

	dataAbstractionSet.SetDefaultValues(int32(numberOfInitialDataAbstractionUnits))

	var dataAbstractionUnitNumber int32 = -1

	for {
		read, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}

		read = strings.TrimSpace(read)
		if len(read) == 0 {
			if err != nil {
				break
			}
			continue
		}

		dataAbstractionUnitNumber++

		if dataAbstractionUnitNumber == numberOfInitialDataAbstractionUnits {
			break
		}

		if dataAbstractionUnitNumber%5000 == 0 && dataAbstractionUnitNumber != 0 {
			fmt.Println("Current number of data abstraction units read:", dataAbstractionUnitNumber, ", more to read...")
		}

		readNumbers := strings.Split(read, ",")

		var classLabelNumber int64
		classLabelNumber, _ = strconv.ParseInt(readNumbers[0], 10, 32)
		if classLabelNumber > int64(maximumClassLabelNumber) {
			panic("Not finished successfully. Not enough colours specified for class label numbers.")
		}

		var dataAbstractionUnit DataAbstraction.DataAbstractionUnit
		dataAbstractionUnit.SetDefaultValues()
		dataAbstractionUnit.ClassLabelNumber = int32(classLabelNumber)

		dataAbstractionUnit.OriginalSpaceCoordinates = make([]float64, len(readNumbers)-1)

		for i := 1; i < len(readNumbers); i++ {
			dataAbstractionUnit.OriginalSpaceCoordinates[i-1], _ = strconv.ParseFloat(readNumbers[i], 64)
		}

		dataAbstractionUnit.DataAbstractionUnitNumber = dataAbstractionUnitNumber

		dataAbstractionSet.DataAbstractionUnits[dataAbstractionUnitNumber] = dataAbstractionUnit

		if err != nil {
			break
		}
	}

	return dataAbstractionSet
}

func WriteEmbeddingToFile(dataAbstractionUnitVisibilitiesToBeShuffled []*DataAbstraction.DataAbstractionUnitVisibility, filePath string, colouring int32, dataAbstractionSet *DataAbstraction.DataAbstractionSet, coloursList []string) {
	numberOfDataAbstractionUnits := int32(len(dataAbstractionUnitVisibilitiesToBeShuffled))
	dataAbstractionUnitVisibilities := make([]*DataAbstraction.DataAbstractionUnitVisibility, numberOfDataAbstractionUnits)
	copy(dataAbstractionUnitVisibilities, dataAbstractionUnitVisibilitiesToBeShuffled)

	randomGenerator := rand.New(rand.NewSource(849662123548415231))
	randomGenerator.Shuffle(len(dataAbstractionUnitVisibilities), func(i, j int) {
		dataAbstractionUnitVisibilities[i], dataAbstractionUnitVisibilities[j] = dataAbstractionUnitVisibilities[j], dataAbstractionUnitVisibilities[i]
	})

	var i, j int32

	var xLow float64 = math.Inf(1)
	var xHigh float64 = math.Inf(-1)

	var yLow float64 = math.Inf(1)
	var yHigh float64 = math.Inf(-1)

	for i = 0; i < numberOfDataAbstractionUnits; i++ {
		dataAbstractionUnitVisibility := dataAbstractionUnitVisibilities[i]

		for j = 0; j < int32(len(dataAbstractionUnitVisibility.VisualSpaceCoordinates)); j++ {
			x := dataAbstractionUnitVisibility.VisualSpaceCoordinates[j][0]
			y := dataAbstractionUnitVisibility.VisualSpaceCoordinates[j][1]

			if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
				panic("Not finished successfully. Unstable floating point calculations.")
			}

			xLow = math.Min(xLow, x)
			xHigh = math.Max(xHigh, x)
			yLow = math.Min(yLow, y)
			yHigh = math.Max(yHigh, y)
		}
	}

	var contextWidth int = 2000
	var contextHeight int = 2000

	multiplicationFactor := (float64(contextWidth) / (xHigh - xLow))
	multiplicationFactor = math.Min(multiplicationFactor, (float64(contextHeight) / (yHigh - yLow)))

	var marginX float64 = 20
	var marginY float64 = 20
	if colouring > 2 {
		marginX = math.Round(float64(dataAbstractionSet.DataAbstractionUnits[0].ImageWidth)/2.0) + 5
		marginY = math.Round(float64(dataAbstractionSet.DataAbstractionUnits[0].ImageHeight)/2.0) + 5
	}

	contextWidth += int(marginX * 2)
	contextHeight += int(marginY * 2)

	colours := make([][3]float64, len(coloursList))

	for i = 0; i < int32(len(coloursList)); i++ {
		colour := coloursList[i]
		if colour[0] != '#' && len(colour) != 7 {
			panic("Not finished successfully.")
		}

		var err error
		var red, green, blue int64
		red, err = strconv.ParseInt(colour[1:3], 16, 16)
		if err != nil {
			panic("Not finished successfully.")
		}

		green, err = strconv.ParseInt(colour[3:5], 16, 16)
		if err != nil {
			panic("Not finished successfully.")
		}

		blue, err = strconv.ParseInt(colour[5:7], 16, 16)
		if err != nil {
			panic("Not finished successfully.")
		}

		colours[i] = [3]float64{float64(red) / 255.0, float64(green) / 255.0, float64(blue) / 255.0}
	}

	context := gg.NewContext(contextWidth, contextHeight)
	context.SetRGB(1, 1, 1)
	context.DrawRectangle(0, 0, float64(contextWidth), float64(contextHeight))
	context.Fill()

	for layerPriority := 0; layerPriority < 2; layerPriority++ {
		for i = 0; i < numberOfDataAbstractionUnits; i++ {
			dataAbstractionUnitVisibility := dataAbstractionUnitVisibilities[i]
			dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[dataAbstractionUnitVisibility.DataAbstractionUnitNumber]

			var redLayerOrNA bool = true
			if dataAbstractionUnitVisibility.Layer == "gray" {
				redLayerOrNA = false
			}

			if colouring != 2 {
				if layerPriority == 0 && redLayerOrNA {
					continue
				}

				if layerPriority == 1 && !redLayerOrNA {
					continue
				}
			} else {
				if layerPriority == 1 && redLayerOrNA {
					continue
				}

				if layerPriority == 0 && !redLayerOrNA {
					continue
				}
			}

			for j = 0; j < int32(len(dataAbstractionUnitVisibility.VisualSpaceCoordinates)); j++ {
				colour := colours[dataAbstractionUnitVisibility.ClassLabelNumber]
				if colouring == 1 {
					if redLayerOrNA {
						colour = [3]float64{0.5, 0, 0}
					} else {
						colour = [3]float64{0.5, 0.5, 0.5}
					}
				}

				context.SetRGB(colour[0], colour[1], colour[2])
				x := dataAbstractionUnitVisibility.VisualSpaceCoordinates[j][0]
				y := dataAbstractionUnitVisibility.VisualSpaceCoordinates[j][1]
				x = (x-xLow)*multiplicationFactor + marginX
				y = (y-yLow)*multiplicationFactor + marginY

				if colouring == 4 {
					if redLayerOrNA {
						if dataAbstractionUnit.ImageRGB == nil {
							image := dataAbstractionUnit.CreateImageGrayscaleRed()
							context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
						} else {
							image := dataAbstractionUnit.CreateImageRGB()
							context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
							context.SetRGB(1, 0, 0)
							context.DrawRectangle(x-float64(dataAbstractionUnit.ImageWidth)/2, y+float64(dataAbstractionUnit.ImageHeight)/2, float64(dataAbstractionUnit.ImageWidth), float64(dataAbstractionUnit.ImageHeight))
							context.SetLineWidth(2.0)
							context.Stroke()
						}
					} else {
						if dataAbstractionUnit.ImageRGB == nil {
							image := dataAbstractionUnit.CreateImageGrayscaleGray()
							context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
						} else {
							image := dataAbstractionUnit.CreateImageRGB()
							context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
							context.SetRGB(0.5, 0.5, 0.5)
							context.DrawRectangle(x-float64(dataAbstractionUnit.ImageWidth)/2, y+float64(dataAbstractionUnit.ImageHeight)/2, float64(dataAbstractionUnit.ImageWidth), float64(dataAbstractionUnit.ImageHeight))
							context.SetLineWidth(2.0)
							context.Stroke()
						}
					}
				} else if colouring == 3 {
					if dataAbstractionUnit.ImageRGB == nil {
						image := dataAbstractionUnit.CreateImageGrayscale()
						context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
					} else {
						image := dataAbstractionUnit.CreateImageRGB()
						context.DrawImage(image, int(x-float64(dataAbstractionUnit.ImageWidth)/2), int(y+float64(dataAbstractionUnit.ImageHeight)/2))
					}

				} else if colouring == 2 {
					if redLayerOrNA {
						context.DrawCircle(x, y, 15)
						context.Fill()
					} else {
						context.DrawCircle(x, y, 15)
						context.Fill()
						context.SetRGB(0, 0, 0)
						context.DrawCircle(x, y, 15)
						context.SetLineWidth(4.0)
						context.Stroke()

						if j == 1 {
							context.DrawCircle(x, y, 5)
							context.Fill()
						}
					}
				} else {
					if redLayerOrNA {
						context.DrawCircle(x, y, 15)
					} else {
						context.DrawCircle(x, y, 10)
					}
					context.Fill()

					if redLayerOrNA {
						context.SetRGB(0, 0, 0)
						context.DrawCircle(x, y, 15)
						context.SetLineWidth(4.0)
						context.Stroke()
					}
				}
			}
		}
	}

	file, _ := os.Create(filePath)
	file.Chmod(Chmod)
	context.EncodePNG(file)
	file.Close()
}

func ReadImagesFileGrayscaleSingleChannel(filePath string, dataAbstractionSet *DataAbstraction.DataAbstractionSet, imageWidth int32, imagesFileHasClassLabelNumbers bool) {
	file, err := os.Open(filePath)

	if err != nil {
		panic("File read error")
	}

	reader := bufio.NewReader(file)

	numberOfDataAbstractionUnits := int32(len(dataAbstractionSet.DataAbstractionUnits))

	var dataAbstractionUnitNumber int32 = -1

	for {
		read, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}

		read = strings.TrimSpace(read)
		if len(read) == 0 {
			if err != nil {
				break
			}
			continue
		}

		dataAbstractionUnitNumber++

		if dataAbstractionUnitNumber == numberOfDataAbstractionUnits {
			break
		}

		readNumbers := strings.Split(read, ",")

		if imagesFileHasClassLabelNumbers {
			readNumbers = readNumbers[1:]
		}

		dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[dataAbstractionUnitNumber]

		if imageWidth == -1 {
			dataAbstractionUnit.ImageWidth = int32(math.Floor(math.Sqrt(float64(len(readNumbers))) + (1e-6)))
		} else {
			dataAbstractionUnit.ImageWidth = imageWidth
		}

		dataAbstractionUnit.ImageHeight = int32(len(readNumbers)) / dataAbstractionUnit.ImageWidth

		if imageWidth == -1 && dataAbstractionUnit.ImageWidth != dataAbstractionUnit.ImageHeight {
			panic("Not finished successfully.")
		}

		if dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight != int32(len(readNumbers)) {
			panic("Not finished successfully.")
		}

		dataAbstractionUnit.ImageGrayscale = make([]uint8, dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight)

		for i := 0; i < len(readNumbers); i++ {
			number, err := strconv.ParseInt(readNumbers[i], 10, 16)
			if err != nil {
				panic("Not finished successfully.")
			}
			dataAbstractionUnit.ImageGrayscale[i] = uint8(number)
		}

		if err != nil {
			break
		}
	}

	if dataAbstractionUnitNumber < numberOfDataAbstractionUnits-1 {
		panic("Not finished successfully")
	}
}

func ReadImagesFileRedGreenBlueChannels(filePath string, dataAbstractionSet *DataAbstraction.DataAbstractionSet, imageWidth int32, imagesFileHasClassLabelNumbers bool) {
	file, err := os.Open(filePath)

	if err != nil {
		panic("File read error")
	}

	reader := bufio.NewReader(file)

	numberOfDataAbstractionUnits := int32(len(dataAbstractionSet.DataAbstractionUnits))

	var dataAbstractionUnitNumber int32 = -1

	for {
		read, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}

		read = strings.TrimSpace(read)
		if len(read) == 0 {
			if err != nil {
				break
			}
			continue
		}

		dataAbstractionUnitNumber++

		if dataAbstractionUnitNumber == numberOfDataAbstractionUnits {
			break
		}

		readNumbers := strings.Split(read, ",")

		if imagesFileHasClassLabelNumbers {
			readNumbers = readNumbers[1:]
		}

		dataAbstractionUnit := &dataAbstractionSet.DataAbstractionUnits[dataAbstractionUnitNumber]

		if imageWidth == -1 {
			dataAbstractionUnit.ImageWidth = int32(math.Floor(math.Sqrt(float64(len(readNumbers))/3.0) + (1e-6)))
		} else {
			dataAbstractionUnit.ImageWidth = imageWidth
		}

		dataAbstractionUnit.ImageHeight = int32(len(readNumbers)) / (dataAbstractionUnit.ImageWidth * 3)

		if imageWidth == -1 && dataAbstractionUnit.ImageWidth != dataAbstractionUnit.ImageHeight {
			panic("Not finished successfully.")
		}

		if dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight*3 != int32(len(readNumbers)) {
			panic("Not finished successfully.")
		}

		dataAbstractionUnit.ImageRGB = make([]uint8, dataAbstractionUnit.ImageWidth*dataAbstractionUnit.ImageHeight*3)

		for i := 0; i < len(readNumbers); i++ {
			number, err := strconv.ParseInt(readNumbers[i], 10, 16)
			if err != nil {
				panic("Not finished successfully.")
			}
			dataAbstractionUnit.ImageRGB[i] = uint8(number)
		}

		if err != nil {
			break
		}
	}

	if dataAbstractionUnitNumber < numberOfDataAbstractionUnits-1 {
		panic("Not finished successfully")
	}
}

func WriteLegendFileHtml(filePath string, classLabels []string, coloursList []string) {
	var radiusBig int32 = 16
	var radiusSmall int32 = 8

	var html strings.Builder
	html.WriteString("<html>\r\n")
	html.WriteString("<head><title></title></head>\r\n")
	html.WriteString(fmt.Sprintf("<style>.legend-entry-circle-big{min-width:%dpx;min-height:%dpx;max-width:%dpx;max-height:%dpx;display:inline-block;border-radius:%dpx;vertical-align:middle;}</style>\r\n", radiusBig*2-4, radiusBig*2-4, radiusBig*2-4, radiusBig*2-4, radiusBig))
	html.WriteString(fmt.Sprintf("<style>.legend-entry-circle-small{min-width:%dpx;min-height:%dpx;max-width:%dpx;max-height:%dpx;display:inline-block;border-radius:%dpx;vertical-align:middle;}</style>\r\n", radiusSmall*2, radiusSmall*2, radiusSmall*2, radiusSmall*2, radiusSmall))
	html.WriteString("<style>.legend-entry-big{min-height:32px;display:inline-block;line-height:32px;margin-left:5px;}</style>\r\n")
	html.WriteString("<style>.legend-entry-small{min-height:32px;display:inline-block;line-height:16px;margin-left:5px;}</style>\r\n")
	html.WriteString("<body style=\"font-size:18px;\">\r\n")

	html.WriteString("<div style=\"border:2px solid black; padding:10px;margin:10px;\">\r\n")
	html.WriteString("Colouring 0 red layer:\r\n")
	for i := 0; i < len(classLabels); i++ {
		html.WriteString("<div style=\"margin-top:10px;margin-bottom:10px;\"><div class=\"legend-entry-circle-big\" style=\"border:2px solid black;background-color:" + coloursList[i] + ";\"></div>")
		html.WriteString("<div class=\"legend-entry-big\">" + classLabels[i] + "</div></div>\r\n")
	}
	html.WriteString("</div>\r\n")

	html.WriteString("<div style=\"border:2px solid black; padding:10px;margin:10px;\">\r\n")
	html.WriteString("<div style=\"margin-bottom:10px;\">Colouring 0 gray layer:</div>\r\n")
	for i := 0; i < len(classLabels); i++ {
		html.WriteString("<div style=\"margin-top:4px;\"><div class=\"legend-entry-circle-small\" style=\"background-color:" + coloursList[i] + ";\"></div>")
		html.WriteString("<div class=\"legend-entry-small\">" + classLabels[i] + "</div></div>\r\n")
	}
	html.WriteString("</div><div style=\"border:2px solid black; padding:10px;margin:10px;\">\r\n")
	html.WriteString("Colouring 2 gray layer:\r\n")
	for i := 0; i < len(classLabels); i++ {
		html.WriteString("<div style=\"margin-top:10px;margin-bottom:10px;\"><div class=\"legend-entry-circle-big\" style=\"border:2px solid black;background-color:" + coloursList[i] + ";\"></div>")
		html.WriteString("<div class=\"legend-entry-big\">" + classLabels[i] + "</div></div>\r\n")
	}

	html.WriteString("<div style=\"margin-top:10px;margin-bottom:10px;\"><div class=\"legend-entry-circle-big\" style=\"border:2px solid black;background-color:white;text-align:center;line-height:28px;font-size:12px;\">⚫</div>")
	html.WriteString("<div class=\"legend-entry-big\">Second projection</div></div>\r\n")

	html.WriteString("</div>\r\n")

	html.WriteString("</body>\r\n")
	html.WriteString("</html>\r\n")

	bytes, _ := ioutil.ReadAll(strings.NewReader(html.String()))
	ioutil.WriteFile(filePath, bytes, Chmod)

}

func WriteShowFileHtml(outputDirectory string, dataAbstractionUnitVisibilities []*DataAbstraction.DataAbstractionUnitVisibility) {
	var html strings.Builder
	html.WriteString(WebUserInterface.WebUserInterfaceCode)

	bytes, _ := ioutil.ReadAll(strings.NewReader(html.String()))
	ioutil.WriteFile(filepath.Join(outputDirectory, "show.html"), bytes, Chmod)
}
