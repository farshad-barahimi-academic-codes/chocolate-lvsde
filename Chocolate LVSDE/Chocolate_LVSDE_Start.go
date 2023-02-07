/*
	This project is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.

	Farshad Barahimi is the only code author for this project including but not limited to this file. All codes in this project including but not limited to this file are written by Farshad Barahimi.

	LVSDE stands for Layered Vertex Splitting Data Embedding.
	For more information about LVSDE dimensionality reduction technique (algorithm) look at the following arXiv preprint:
	Farshad Barahimi, "Multi-point dimensionality reduction to improve projection layout reliability",  arXiv:2101.06224v5, 2022.

	The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

	Copyright notice for this code (this implementation of LVSDE):
	Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.

	"Chocolate LVSDE" is the name of this implementation of LVSDE.

	The purpose of writing this code is academic.

	While this project sometimes uses UMAP and for comparison t-SNE, the author is not the author of UMAP or t-SNE and the copyright notice above does not correspond to implementations of UMAP and t-SNE in umap-learn library or scikit-learn library.
*/
package main

import "C"
import (
	"fmt"
	"github.com/farshad-barahimi-academic-codes/chocolate-lvsde/EmbeddingSpecification"
	"os"
)

/*
	Compilation guide:
	To compile make sure the following environmental variables are set correctly:
	CGO_ENABLED, CGO_CFLAGS, CGO_LDFLAGS and CC

		Example for Windows:
		CGO_ENABLED=1
		CGO_CFLAGS=-IC:/Users/TheUser/miniconda3/envs/temp-env/include -DMS_WIN64
		CGO_LDFLAGS=-LC:/Users/TheUser/miniconda3/envs/temp-env/libs -lpython310 -lpthread -lm
		CC=gcc

		Example for Linux:
		CGO_ENABLED=1
		CGO_CFLAGS=-I/usr/include/python3.10
		CGO_LDFLAGS=-L/usr/lib/x86_64-linux-gnu -l:libpython3.10.a -lpthread -lm -lexpat -lz
		CC=x86_64-linux-gnu-gcc-11

	Run guide:
	To run make sure PYTHONHOME environmental variable in set correctly and umap-learn Python library is installed.
*/
func main() {

	fmt.Println("Chocolate LVSDE 1.16")
	fmt.Println("An implementation of the Layered Vertex Splitting Data Embedding (LVSDE) dimensionality reduction technique.")
	fmt.Println("Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.")
	fmt.Println("")

	args := os.Args

	if len(args) == 1 {
		fmt.Println("Please specify the absolute path for the embedding specifications file as the only argument or --structure-help for information on the structure of embedding specification file.")
	} else if len(args) == 2 && args[1] != "--structure-help" {
		embeddingSpecificationsFilePath := args[1]
		embeddingSpecifications := EmbeddingSpecification.ReadEmbeddingSpecification(embeddingSpecificationsFilePath)
		if len(embeddingSpecifications.EmbeddingSpecifications) > 1 {
			panic("Not finished successfully")
		}
		EmbeddingSpecification.RunEmbeddingSpecifications(embeddingSpecifications.EmbeddingSpecifications[0:1])
	} else if len(args) == 2 && args[1] == "--structure-help" {
		fmt.Println("Structure of embedding specification file which is a JSON file where some of the JSON elements have default values and paths are relative to the directory containing the embedding specification file:")
		fmt.Println("{\n\t\"embedding_specifications\":[\n\t{\n\t\t\"input_file_path\":\"\",\n\t\t\"output_directory\":\"\",\n\t\t\"is_input_file_distances\":\"\",\n\t\t\"number_of_initial_data_abstraction_units\":\"\",\n\t\t\"visual_density_adjustment_parameter\":\"\",\n\t\t\"number_of_neighbours_for_building_neighbourhood_graph\":\"\",\n\t\t\"evaluation_neighbourhood_sizes\":[],\n\t\t\"preliminary_to_thirty_dimensions_umap\":\"\",\n\t\t\"random_state\":\"\",\n\t\t\"class_labels\":[],\n\t\t\"compare_with_other_methods\":\"\",\n\t\t\"colours_list\":\"\",\n\t\t\"images_file_red_green_blue_channels\":\"\",\n\t\t\"images_file_grayscale_single_channel\":\"\",\n\t\t\"images_file_image_width\":\"\",\n\t\t\"images_file_has_class_label_numbers\":\"\",\n\t\t\"random_seed\":\"\",\n\t\t\"preliminary_to_thirty_dimensions_umap\":\"\",\n\t\t\"number_of_secondary_data_abstraction_units\":\"\",\n\t\t\"use_cosine_distance_for_input_multi_dimensional_data\":\"\"\n\t}\n\t]\n}")
	} else {
		fmt.Println("Incorrect number of arguments")
	}
}
