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

package FileReadingOrWriting

import (
	"bufio"
	"fmt"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"
	"io"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

import "github.com/fogleman/gg"

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

func WriteEmbeddingToFile(dataAbstractionUnitVisibilities []*DataAbstraction.DataAbstractionUnitVisibility, filePath string, colouring int32, dataAbstractionSet *DataAbstraction.DataAbstractionSet, coloursList []string) {
	var numberOfDataAbstractionUnits int32 = int32(len(dataAbstractionUnitVisibilities))
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
				x = (x - xLow) * multiplicationFactor
				y = (y - yLow) * multiplicationFactor

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

	context.SavePNG(filePath)
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
	html.WriteString("<body>\r\n")

	html.WriteString("<div style=\"border:2px solid black; padding:10px;margin:10px;\">\r\n")
	html.WriteString("Colouring 0 red layer:\r\n")
	for i := 0; i < len(classLabels); i++ {
		html.WriteString("<div><div class=\"legend-entry-circle-big\" style=\"border:2px solid black;background-color:" + coloursList[i] + ";\"></div>")
		html.WriteString("<div class=\"legend-entry-big\">" + classLabels[i] + "</div></div>\r\n")
	}
	html.WriteString("</div>\r\n")

	html.WriteString("<div style=\"border:2px solid black; padding:10px;margin:10px;\">\r\n")
	html.WriteString("Colouring 0 gray layer:\r\n")
	for i := 0; i < len(classLabels); i++ {
		html.WriteString("<div><div class=\"legend-entry-circle-small\" style=\"background-color:" + coloursList[i] + ";\"></div>")
		html.WriteString("<div class=\"legend-entry-small\">" + classLabels[i] + "</div></div>\r\n")
	}
	html.WriteString("<div><div class=\"legend-entry-small\">⊙ Second projection</div></div>\r\n")
	html.WriteString("</div>\r\n")

	html.WriteString("</body>\r\n")
	html.WriteString("</html>\r\n")

	bytes, _ := ioutil.ReadAll(strings.NewReader(html.String()))
	ioutil.WriteFile(filePath, bytes, 640)

}

func WriteShowFileHtml(outputDirectory string, dataAbstractionUnitVisibilities []*DataAbstraction.DataAbstractionUnitVisibility) {
	jsCode := `/*
	Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.
	The purpose of writing this code is academic.
*/

document.addEventListener("DOMContentLoaded", function(event) {
	document.querySelector("#show-button").addEventListener('click', function() {
		if(document.querySelector("#file-input").value == '') {
			return;
		}
		
		const log=document.querySelector("#log")
		log.innerHTML='Please wait...'

		var file = document.querySelector("#file-input").files[0];

		var fileReader = new FileReader();
		fileReader.onload = function(e) {
			readData(e.target.result)
		};
		fileReader.onerror = function(e) {
			throw new Error("Error")
		};
		fileReader.readAsArrayBuffer(file);
	});

	document.querySelector("#canvas").addEventListener('mousedown', e => {
		if(e.which==1){
			let x = e.offsetX + document.querySelector("#canvas").getBoundingClientRect().left;
			let y = e.offsetY + document.querySelector("#canvas").getBoundingClientRect().top;
			const blue=document.querySelector("#blue")
			blue.style.left=x.toString()+'px'
			blue.style.top=y.toString()+'px'
			blue.style.width='0px'
			blue.style.height='0px'
			if(blue.style.display!='block')
				blue.style.display='block'
		}
	});

	document.querySelector("#canvas").addEventListener('mousemove', e => {
		if(e.which==1){
			let x = e.offsetX + document.querySelector("#canvas").getBoundingClientRect().left;
			let y = e.offsetY + document.querySelector("#canvas").getBoundingClientRect().top;
			const blue=document.querySelector("#blue")
			let width= Math.max(0, x- blue.getBoundingClientRect().left)
			let height= Math.max(0, y- blue.getBoundingClientRect().top)
			blue.style.width=width.toString()+'px'
			blue.style.height=height.toString()+'px'
		}
	});

	document.querySelector("#canvas").addEventListener('mouseup', e => {
		if(e.which==1){
			const blue=document.querySelector("#blue")
			let x=blue.getBoundingClientRect().left - document.querySelector("#canvas").getBoundingClientRect().left
			let y=blue.getBoundingClientRect().top - document.querySelector("#canvas").getBoundingClientRect().top
			let width=blue.clientWidth
			let height=blue.clientHeight
			document.querySelector("#log").innerHTML='Please wait...'
			updateBasedOnSelection(x,y,width,height)
			document.querySelector("#log").innerHTML='(x:'+x.toString()+', y:'+y.toString()+', w:'+width.toString()+', h:'+height.toString()+')'
		}
	});
});

let iterationsData=[]
let coloursList=[]
let currentIteration=1829
let imagesWidth=0
let imagesHeight=0
let images=[]
let imagesRed=[]
let imagesGray=[]
let numberOfDataAbstractionUnits=0
let colouringOption=0
let sizeOption=0
let overlapReductionRounds=20

function readData(bsonFileContentZipped){
	
	var new_zip = new JSZip();
	new_zip.loadAsync(new Uint8Array(bsonFileContentZipped)).then(function(zip) {
		zip.file("embedding_details.bson").async("uint8array",function updateCallback(metadata) {
			document.querySelector("#log").innerHTML='Please wait... ' + (metadata.percent/2).toFixed(1) + '%';}).then(function (bsonFileContent) {
			const embeddingDetails = BSON.deserialize(bsonFileContent);
    
			console.log('embeddingDetails:', embeddingDetails);
			iterationsData=[]
			coloursList=embeddingDetails['colours_list']
			let imagesGrayscale=embeddingDetails['images_grayscale_single_channel']
			let imagesRedGreenBlueChannels=embeddingDetails['images_red_green_blue_channels']
			imagesWidth=embeddingDetails['image_width']
			numberOfDataAbstractionUnits=embeddingDetails['embedding_iterations'][0].length
			for (let iteration=0;iteration<1830;iteration++){
				iterationsData.push([])
				let iterationData=embeddingDetails['embedding_iterations'][iteration]
				for(let i=0;i<iterationData.length;i++){
					for(let j=0;j<iterationData[i].v.length;j++){
						iterationsData[iteration].push([iterationData[i].v[j][0],iterationData[i].v[j][1],20,20,iterationData[i].l,iterationData[i].c,iterationData[i].d,j,false,false,0,0])
					}
				}
			}

			console.log('iterationsData:',iterationsData)

			if(imagesGrayscale.length>0){
				for(let i=0;i<numberOfDataAbstractionUnits;i++){
					let flat=imagesGrayscale[i].value()
					if(i==0)
						imagesHeight= flat.length / imagesWidth
					let rgba=[]
					let rgbaRed=[]
					let rgbaGray=[]
					for(let j=0;j<imagesHeight;j++){
						for(let k=0;k<imagesWidth;k++){
							let grayscale=flat[j*imagesWidth+k].charCodeAt(0)
							rgba.push(grayscale)
							rgba.push(grayscale)
							rgba.push(grayscale)
							rgba.push(255)
							rgbaRed.push(grayscale)
							rgbaRed.push(0)
							rgbaRed.push(0)
							rgbaRed.push(255)
							rgbaGray.push(Math.floor(127 + grayscale*128/255))
							rgbaGray.push(Math.floor(127 + grayscale*128/255))
							rgbaGray.push(Math.floor(127 + grayscale*128/255))
							rgbaGray.push(255)
						}
					}
					
					images.push(new ImageData(Uint8ClampedArray.from(rgba),imagesWidth,imagesHeight))
					imagesRed.push(new ImageData(Uint8ClampedArray.from(rgbaRed),imagesWidth,imagesHeight))
					imagesGray.push(new ImageData(Uint8ClampedArray.from(rgbaGray),imagesWidth,imagesHeight))
				}
				colouringOption=4
			}
			else if(imagesRedGreenBlueChannels.length>0){
				for(let i=0;i<numberOfDataAbstractionUnits;i++){
					let flat=imagesRedGreenBlueChannels[i].value()
					if(i==0)
						imagesHeight= (flat.length / imagesWidth) / 3
					let rgba=[]
					let rgbaRed=[]
					let rgbaGray=[]
					for(let j=0;j<imagesHeight;j++){
						for(let k=0;k<imagesWidth;k++){
							let red=flat[j*imagesWidth+k].charCodeAt(0)
							let green=flat[imagesWidth*imagesHeight+j*imagesWidth+k].charCodeAt(0)
							let blue=flat[2*imagesWidth*imagesHeight+j*imagesWidth+k].charCodeAt(0)
							rgba.push(red)
							rgba.push(green)
							rgba.push(blue)
							rgba.push(255)
							rgbaRed.push(red)
							rgbaRed.push(green)
							rgbaRed.push(blue)
							rgbaRed.push(255)
							rgbaGray.push(red)
							rgbaGray.push(green)
							rgbaGray.push(blue)
							rgbaGray.push(255)
						}
					}
					
					images.push(new ImageData(Uint8ClampedArray.from(rgba),imagesWidth,imagesHeight))
					imagesRed.push(new ImageData(Uint8ClampedArray.from(rgbaRed),imagesWidth,imagesHeight))
					imagesGray.push(new ImageData(Uint8ClampedArray.from(rgbaGray),imagesWidth,imagesHeight))
				}
				colouringOption=4
			}

			showData()
		});
	});
}

function updateBasedOnSelection(x,y,width,height){
	let data=iterationsData[currentIteration]
	for(let i=0;i<data.length;i++){
		data[i][9]=false
	}

	for(let i=0;i<data.length;i++){
		if(data[i][10]>=x && data[i][10]<=x+width && data[i][11]>=y && data[i][11]<=y+height){
			data[i][8]=true
			for(let j=0;j<data.length;j++){
				if(i!=j && data[i][6]==data[j][6]){
					data[j][9]=true
				}
			}
		}
		else{
			data[i][8]=false
		}
	}

	showIteration(currentIteration)
}

function showData(){
	showIteration(1829)
	
	const blue=document.querySelector("#blue")
	blue.style.display='none'
	blue.style.backgroundColor='rgba(255,255,255,0)'
	blue.style.border='2px solid blue'
	blue.style.top='0px'
	blue.style.left='0px'
	blue.style.width='0px'
	blue.style.height='0px'

	logCurrentIteration()
	
	document.querySelector("#show-button").style.display='none'
	document.querySelector("#file-input").style.display='none'
	document.querySelector("#prev-button").style.display='inline-block'
	document.querySelector("#next-button").style.display='inline-block'
	document.querySelector("#last-button").style.display='inline-block'
	document.querySelector("#first-button").style.display='inline-block'
	document.querySelector("#prev-100-button").style.display='inline-block'
	document.querySelector("#next-100-button").style.display='inline-block'
	document.querySelector("#change-colouring-button").style.display='inline-block'
	document.querySelector("#reduce-overlap-button").style.display='inline-block'
	document.querySelector("#change-sizing-button").style.display='inline-block'
	document.querySelector("#change-overlap-reduction-rounds-button").style.display='inline-block'

	document.querySelector("#prev-button").addEventListener('click', function() {
		if(currentIteration>0){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(currentIteration-1);
			logCurrentIteration()
		}
	});

	document.querySelector("#next-button").addEventListener('click', function() {
		if(currentIteration<1829){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(currentIteration+1);
			logCurrentIteration()
		}
	});

	document.querySelector("#last-button").addEventListener('click', function() {
		if(currentIteration<1829){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(1829);
			logCurrentIteration()
		}
	});

	document.querySelector("#first-button").addEventListener('click', function() {
		if(currentIteration>0){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(0);
			logCurrentIteration()
		}
	});

	document.querySelector("#prev-100-button").addEventListener('click', function() {
		if(currentIteration>0){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(Math.max(currentIteration-100,0));
			logCurrentIteration()
		}
	});

	document.querySelector("#next-100-button").addEventListener('click', function() {
		if(currentIteration<1829){
			document.querySelector("#log").innerHTML='Please wait...'
			showIteration(Math.min(currentIteration+100,1829));
			logCurrentIteration()
		}
	});

	document.querySelector("#change-colouring-button").addEventListener('click', function() {
		if(images.length>0)
			colouringOption=(colouringOption+1)%5
		else
			colouringOption=(colouringOption+1)%3

		document.querySelector("#log").innerHTML='Please wait...'
		showIteration(currentIteration);
		logCurrentIteration()
	});

	document.querySelector("#reduce-overlap-button").addEventListener('click', function() {
		document.querySelector("#prev-button").style.display='none'
		document.querySelector("#next-button").style.display='none'
		document.querySelector("#last-button").style.display='none'
		document.querySelector("#first-button").style.display='none'
		document.querySelector("#prev-100-button").style.display='none'
		document.querySelector("#next-100-button").style.display='none'
		document.querySelector("#change-colouring-button").style.display='none'
		document.querySelector("#reduce-overlap-button").style.display='none'
		document.querySelector("#change-sizing-button").style.display='none'
		document.querySelector("#change-overlap-reduction-rounds-button").style.display='none'
		reduceOverlap(overlapReductionRounds);
	});

	document.querySelector("#change-sizing-button").addEventListener('click', function() {
		sizeOption=(sizeOption+1)%4

		document.querySelector("#log").innerHTML='Please wait...'
		showIteration(currentIteration);
		logCurrentIteration()
	});

	document.querySelector("#change-overlap-reduction-rounds-button").addEventListener('click', function() {
		overlapReductionRounds=(overlapReductionRounds+1)%4

		document.querySelector("#log").innerHTML='Please wait...'
		showIteration(currentIteration);
		logCurrentIteration()
	});
}

function logCurrentIteration(){
	if(currentIteration==1829){
		document.querySelector("#log").innerHTML='Iteration: last iteration'
	} else {
		document.querySelector("#log").innerHTML='Iteration: '+(currentIteration+1).toString()
	}
}

function showIteration(iteration) {
	currentIteration=iteration
	showDataOnCanvas(iterationsData[iteration],null)
}

function showDataOnCanvas(data,canvas) {
	let xLow  = Number.POSITIVE_INFINITY
	let xHigh = Number.NEGATIVE_INFINITY

	let yLow = Number.POSITIVE_INFINITY
	let yHigh = Number.NEGATIVE_INFINITY

	for(let i = 0; i < data.length; i++) {
		let x = data[i][0]
		let y = data[i][1]

		if(canvas!==null){
			x=data[i][10]
			y=data[i][11]
		}

		xLow = Math.min(xLow, x)
		xHigh = Math.max(xHigh, x)
		yLow = Math.min(yLow, y)
		yHigh = Math.max(yHigh, y)
		
	}

	let insideMargin=15
	if(images.length>0)
		insideMargin=imagesWidth/2+5

	let sizes=[1000,2000,750,1500]
	let contextWidth = sizes[sizeOption]+insideMargin*2
	let contextHeight = sizes[sizeOption]+insideMargin*2

	let multiplicationFactor = (contextWidth / (xHigh - xLow))
	multiplicationFactor = Math.min(multiplicationFactor, (contextHeight) / (yHigh - yLow))

	let popup=true

	if(canvas===null){
		popup=false
		canvas = document.querySelector("#canvas")
	}
	else {
		contextWidth=xHigh-xLow+insideMargin*2
		contextHeight=yHigh-yLow+insideMargin*2
	}
		
	canvas.style.width=contextWidth.toString() +'px'
	canvas.style.height=contextHeight.toString()+'px'
	canvas.style.border='1px solid black'
	canvas.style.display='block'
	canvas.style.margin='20px'
	canvas.width=contextWidth
	canvas.height=contextHeight
	

	const context=canvas.getContext('2d');
	context.clearRect(0, 0, canvas.width, canvas.height);

	for (let round=1;round<=3;round++){
		for (let i=0;i<data.length;i++){
			let x=data[i][0]
			let y=data[i][1]

			if(popup){
				x=data[i][10]-xLow + insideMargin
				y=data[i][11]-yLow + insideMargin
			}
			else{
				x = (x - xLow) * multiplicationFactor + insideMargin
				y = (y - yLow) * multiplicationFactor + insideMargin
				data[i][10]=x
				data[i][11]=y
			}
			
			let ind=data[i][6]
	
			if(colouringOption==2 && data[i][4]=='gray' && round==1)
				continue;

			if(colouringOption==2 && data[i][4]=='red' && round==2)
				continue;

			if((colouringOption==0 || colouringOption==1 || colouringOption==4) && data[i][4]=='gray' && round==2)
				continue;

			if((colouringOption==0 || colouringOption==1 || colouringOption==4) && data[i][4]=='red' && round==1)
				continue;
			
			if(colouringOption==3 && round==2)
				continue;
	
			if(colouringOption==4){
				if(round!=3){
					if(data[i][4]=='gray')
						context.putImageData(imagesGray[ind],x-imagesWidth/2,y-imagesHeight/2)
					else
						context.putImageData(imagesRed[ind],x-imagesWidth/2,y-imagesHeight/2)
				}
				
				if(data[i][9] && round==3){
					context.strokeStyle = 'yellow';
					context.rect(x-imagesWidth/2-1,y-imagesHeight/2-1,imagesWidth+2,imagesHeight+2)
					context.lineWidth = 2;
					context.stroke();
				}
			}
			else if(colouringOption==3){
				if(round!=3)
					context.putImageData(images[ind],x-imagesWidth/2,y-imagesHeight/2)
				
				if(data[i][9] && round==3){
					context.strokeStyle = 'yellow';
					context.rect(x-imagesWidth/2-1,y-imagesHeight/2-1,imagesWidth+2,imagesHeight+2)
					context.lineWidth = 2;
					context.stroke();
				}
			}
			else if(colouringOption==2){
				let radius=10
				if(round!=3){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					context.fillStyle = coloursList[data[i][5]];
					context.fill();
					context.lineWidth = 2;
					
					if(data[i][4]=='gray'){
						context.strokeStyle = 'black';
						context.stroke();
						if(data[i][7]==1){
							context.closePath();
							context.beginPath();
							context.arc(x, y, 5, 0, 2 * Math.PI);
							context.fillStyle = 'black';
							context.fill();
						}
					}
					
					context.closePath();
				} 
				else if(data[i][9]){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					context.fillStyle = 'none'
					context.lineWidth = 2;
					context.strokeStyle = 'yellow';
					context.stroke();
				}
			}
			else if(colouringOption==0){
				let radius=10
				if(data[i][4]=='gray')
					radius=6;
				if(round!=3){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					context.fillStyle = coloursList[data[i][5]];
					context.fill();
					context.lineWidth = 2;
					if(data[i][4]=='red'){
						context.strokeStyle = 'black';
						context.stroke();
					}
					
					context.closePath();
				}
				else if(data[i][9]){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					context.fillStyle = 'none'
					context.lineWidth = 2;
					context.strokeStyle = 'yellow';
					context.stroke();
				}
			}
			else if(colouringOption==1){
				let radius=10
				if(data[i][4]=='gray')
					radius=6;

				if(round!=3){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					if(data[i][4]=='gray')
						context.fillStyle = 'gray';
					else
						context.fillStyle = 'red';
					context.fill();
					context.lineWidth = 2;
					
					context.closePath();
				}
				else if(data[i][9]){
					context.beginPath();
					context.arc(x, y, radius, 0, 2 * Math.PI);
					context.fillStyle = 'none'
					context.lineWidth = 2;
					context.strokeStyle = 'yellow';
					context.stroke();
				}
			}
		}
	}
}

function flatten(toBeFlattened,tree,isRoot){
	let treeIndex1=toBeFlattened[0]
	let flattened=[treeIndex1]
	
	for(let i=1;i<toBeFlattened.length;i++){
		let subFlattened;
		if(!Array.isArray(toBeFlattened[i]))
			subFlattened=[toBeFlattened[i]];
		else
			subFlattened=flatten(toBeFlattened[i],tree,isRoot);
		
		let treeIndex2=subFlattened[0]
		tree[treeIndex1].push(treeIndex2)
		isRoot[treeIndex2]=false
		for(let j=0;j<subFlattened.length;j++)
			flattened.push(subFlattened[j])
	}
	
	return flattened
}

function move(dataCopy,i,j){
	let x1=dataCopy[i][10]
	let y1=dataCopy[i][11]
	let width1=dataCopy[i][2]
	let height1=dataCopy[i][3]
	let left1=x1-width1/2
	let top1=y1-height1/2

	let x2=dataCopy[j][10]
	let y2=dataCopy[j][11]
	let width2=dataCopy[j][2]
	let height2=dataCopy[j][3]
	let left2=x2-width2/2
	let top2=y2-height2/2

	let horizontalDifference;
	let verticalDifference;
	if(left1<=left2)
		horizontalDifference=left2-(left1+width1);
	else
		horizontalDifference=left1-(left2+width2);

	if(top1<=top2)
		verticalDifference=top2-(top1+height1);
	else
		verticalDifference=top1-(top2+height2);

	let t,x,y;

	if(horizontalDifference>=0 && verticalDifference>=0){
		t=Math.sqrt(Math.pow(horizontalDifference,2)+Math.pow(verticalDifference,2))/Math.sqrt(Math.pow(Math.abs(x2-x1), 2)+Math.pow(Math.abs(y2-y1), 2));
		x=x1+(x2-x1)*t
		y=y1+(y2-y1)*t
	}
	else if(horizontalDifference>=0 && verticalDifference<0){
		t=horizontalDifference/(Math.abs(x2-x1)*2.0);
		x=x1+(x2-x1)*t
		y=y1+(y2-y1)*t
	}
	else if(horizontalDifference<0 && verticalDifference>=0){
		t=verticalDifference/(Math.abs(y2-y1)*2.0);
		x=x1+(x2-x1)*t
		y=y1+(y2-y1)*t
	}
	else
	{
		if(Math.abs(x2-x1)>1e-10 && Math.abs(y2-y1)>1e-10){
			t=Math.sqrt(Math.pow((width1 + width2)/2.0, 2)+Math.pow((height1 + height2)/2.0, 2))/Math.sqrt(Math.pow(Math.abs(x2-x1), 2)+Math.pow(Math.abs(y2-y1), 2));
			x=x1+(x2-x1)*t
			y=y1+(y2-y1)*t
		}
		else if(Math.abs(x2-x1)>1e-10){
			t=(width1 + width2)/(Math.abs(x2-x1)*2.0);
			x=x1+(x2-x1)*t
			y=y1+(y2-y1)*t
		}
		else if(Math.abs(y2-y1)>1e-10){
			t=(height1 + height2)/(Math.abs(y2-y1)*2.0);
			x=x1+(x2-x1)*t
			y=y1+(y2-y1)*t
		}
		else {
			if(width1+width2<height1+height2){
				x=x1+(width1+width2)/2.0
				y=y1
			}
			else{
				x=x1
				y=y1+(height1+height2)/2.0
			}
		}
	}

	dataCopy[j][10]=x
	dataCopy[j][11]=y
}

function moveTree(tree,root,dataCopy){
	for(let i=0;i<tree[root].length;i++){
		move(dataCopy,root,tree[root][i])
		moveTree(tree,tree[root][i],dataCopy)
	}
}

function overlapWeight(x1,y1,left1,top1,width1,height1,x2,y2,left2,top2,width2,height2){
	let horizontalDifference;
	let verticalDifference;
	if(left1<=left2)
		horizontalDifference=left2-(left1+width1);
	else
		horizontalDifference=left1-(left2+width2);

	if(top1<=top2)
		verticalDifference=top2-(top1+height1);
	else
		verticalDifference=top1-(top2+height2);

	if(horizontalDifference>=0 && verticalDifference>=0)
		return Math.sqrt(Math.pow(horizontalDifference,2)+Math.pow(verticalDifference,2));
	else if(horizontalDifference>=0 && verticalDifference<0)
		return horizontalDifference;
	else if(horizontalDifference<0 && verticalDifference>=0)
		return verticalDifference;
	else
	{
		let t;
		if(Math.abs(x2-x1)>1e-10 && Math.abs(y2-y1)>1e-10)
			t=Math.sqrt(Math.pow((width1 + width2)/2.0, 2)+Math.pow((height1 + height2)/2.0, 2))/Math.sqrt(Math.pow(Math.abs(x2-x1), 2)+Math.pow(Math.abs(y2-y1), 2));
		else if(Math.abs(x2-x1)>1e-10)
			t=(width1 + width2)/(Math.abs(x2-x1)*2.0);
		else if(Math.abs(y2-y1)>1e-10)
			t=(height1 + height2)/(Math.abs(y2-y1)*2.0);
		else
			return -Math.sqrt(Math.pow((width1 + width2)/2.0, 2)+Math.pow((height1 + height2)/2.0, 2));

		return Math.sqrt(Math.pow(Math.abs(x2-x1), 2)+Math.pow(Math.abs(y2-y1), 2))-t*Math.sqrt(Math.pow(Math.abs(x2-x1), 2)+Math.pow(Math.abs(y2-y1), 2));
	}
}

function reduceOverlap(rounds){
	// The algorithm used for overlap reduction is a modification of the algorithm in the following paper:
	// Nachmanson, Lev, Arlind Nocaj, Sergey Bereg, Leishi Zhang, and Alexander Holroyd. "Node overlap removal by growing a tree." In International Symposium on Graph Drawing and Network Visualization, pp. 33-43. Springer, Cham, 2016.
	let data=iterationsData[currentIteration]
	let dataCopy=[]
	for(let i=0;i<data.length;i++){
		if(data[i][8])
			dataCopy.push(data[i].slice())
	}

	for(let i=0;i<rounds;i++)
		reduceOverlapMove(dataCopy)

	let popup=document.querySelector("#overlap-reduced")

	popup.style.background='white'
	popup.style.border='2px solid black'
	popup.style.left='100px'
	popup.style.top='100px'
	popup.style.width='500px'
	popup.style.height='500px'
	popup.style.pointerEvents='auto'
	popup.style.overflow='scroll'
	popup.style.display='block'

	let bar=document.createElement('div')
	bar.innerHTML='Overlap reduced'
	bar.style.background='#BBBBBB'
	bar.style.textAlign='center'
	bar.style.cursor='default'
	bar.style.padding='3px'
	
	let canvasPopup=document.createElement('canvas')
	canvasPopup.style.border='1px solid black'
	popup.replaceChildren(bar,canvasPopup)
	showDataOnCanvas(dataCopy,canvasPopup)
}

function reduceOverlapMove(dataCopy){
	// The algorithm used for overlap reduction is a modification of the algorithm in the following paper:
	// Nachmanson, Lev, Arlind Nocaj, Sergey Bereg, Leishi Zhang, and Alexander Holroyd. "Node overlap removal by growing a tree." In International Symposium on Graph Drawing and Network Visualization, pp. 33-43. Springer, Cham, 2016.

	let dataCopy1=[]
	
	for (let i=0;i<dataCopy.length;i++){
		let x=dataCopy[i][10];
		let y=dataCopy[i][11];
		dataCopy1.push([x,y])
	}
	
	const delaunay = d3.Delaunay.from(dataCopy1);
	let delaunayPoints=delaunay.points
	let delaunayTriangles=delaunay.triangles

	let weights=[]
	const {MinQueue} = Heapify;
	const weightsQueue = new MinQueue(dataCopy.length*8+10,[],[], Uint32Array, Float64Array);

	
	for (let i=0;i<dataCopy.length;i++){
		let x1=dataCopy[i][10]
		let y1=dataCopy[i][11]
		let width1=dataCopy[i][2]
		let height1=dataCopy[i][3]
		
		let delaunayPointIndex=delaunay.find(x1,y1);
		let neighbourIndices=delaunay.neighbors(i)
		for(let neighbourIndex of neighbourIndices){
			let x2=dataCopy[neighbourIndex][10]
			let y2=dataCopy[neighbourIndex][11]
			let width2=dataCopy[neighbourIndex][2]
			let height2=dataCopy[neighbourIndex][3]
			let weight=overlapWeight(x1,y1,x1-width1/2,y1-height1/2,width1,height1,x2,y2,x2-width2/2,y2-height2/2,width2,height2)
			weightsQueue.push(weights.length,weight)
			weights.push([i,neighbourIndex,weight])
		}
	}

	let unionFind=[]
	let tree=[]
	let isSubFlattened=[]
	let isMerged=[]
	let isRoot=[]
	for (let i=0;i<dataCopy.length;i++){
		unionFind.push([i])
		tree.push([])
		isSubFlattened.push(false)
		isMerged.push(false)
		isRoot.push(true)
	}
	
	while(true){
		let index=weightsQueue.pop()
		if (index===undefined){
			break;
		}
		
		let min=Math.min(weights[index][0],weights[index][1])
		let max=Math.max(weights[index][0],weights[index][1])
		if(min===max)
			throw new Error("Error")
		
		if(isMerged[max])
			continue;

		unionFind[min].push(unionFind[max])
		isMerged[max]=true
	}

	for(let i=0;i<unionFind.length;i++)
		if(isMerged[i])
			unionFind[i]=[]

	for (let i=0;i<dataCopy.length;i++){
		if( unionFind[i].length!=0 && isRoot[i]!==false){
			unionFind[i]=flatten(unionFind[i],tree,isRoot);
		}
	}

	for (let i=0;i<dataCopy.length;i++){
		if( isRoot[i]){
			moveTree(tree,i,dataCopy)
		}
	}
}
`

	var html strings.Builder
	html.WriteString("<!-- Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license. -->")
	html.WriteString("<!-- The purpose of writing this code is academic. -->")
	html.WriteString("<html>\r\n")
	html.WriteString("<head>\r\n")
	html.WriteString("<title></title>\r\n")
	html.WriteString("<script src=\"https://unpkg.com/bson@4.6.3/dist/bson.bundle.js\"></script>\r\n")
	html.WriteString("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\r\n")
	html.WriteString("<script src=\"https://unpkg.com/heapify@0.6.0/heapify.js\"></script>\r\n")
	html.WriteString("<script src=\"https://unpkg.com/jszip@3.9.1/dist/jszip.min.js\"></script>\r\n")
	html.WriteString("<script type=\"module\">\r\n")
	html.WriteString(jsCode)
	html.WriteString("</script>\r\n")
	html.WriteString("</head>\r\n")
	html.WriteString("<body>\r\n")
	html.WriteString("<div style=\"padding:5px;background:#BBBBBB;border-radius:5px;\">\r\n")
	html.WriteString("<input type=\"file\" id=\"file-input\" />\r\n")
	html.WriteString("<button id=\"show-button\">Show</button>\r\n")
	html.WriteString("<span id=\"log\" style=\"margin-left:5px;display:inline-block;min-width:200px;\"></span>\r\n")
	html.WriteString("<button id=\"prev-button\" style=\"display:none;\">Previous</button>\r\n")
	html.WriteString("<button id=\"next-button\" style=\"display:none;\">Next</button>\r\n")
	html.WriteString("<button id=\"last-button\" style=\"display:none;\">Last</button>\r\n")
	html.WriteString("<button id=\"first-button\" style=\"display:none;\">First</button>\r\n")
	html.WriteString("<button id=\"prev-100-button\" style=\"display:none;\">Previous 100</button>\r\n")
	html.WriteString("<button id=\"next-100-button\" style=\"display:none;\">Next 100</button>\r\n")
	html.WriteString("<button id=\"change-colouring-button\" style=\"display:none;\">Change colouring</button>\r\n")
	html.WriteString("<button id=\"reduce-overlap-button\" style=\"display:none;\">Reduce overlap</button>\r\n")
	html.WriteString("<button id=\"change-sizing-button\" style=\"display:none;\">Change sizing</button>\r\n")
	html.WriteString("<button id=\"change-overlap-reduction-rounds-button\" style=\"display:none;\">Change overlap reduction rounds</button>\r\n")
	html.WriteString("</div>\r\n")
	html.WriteString("<div>\r\n")
	html.WriteString("<canvas id=\"canvas\" style=\"position:absolute;\"></canvas>\r\n")
	html.WriteString("<div id=\"blue\" style=\"position:absolute;pointer-events: none;\"></div>\r\n")
	html.WriteString("<div id=\"overlap-reduced\" style=\"position:absolute;pointer-events: none;\"></div>\r\n")
	html.WriteString("</div>\r\n")
	html.WriteString("</body>\r\n")
	html.WriteString("</html>\r\n")

	bytes, _ := ioutil.ReadAll(strings.NewReader(html.String()))
	ioutil.WriteFile(filepath.Join(outputDirectory, "show.html"), bytes, 640)
}
