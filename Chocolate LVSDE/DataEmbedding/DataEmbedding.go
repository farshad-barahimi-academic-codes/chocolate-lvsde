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

import "github.com/farshad-barahimi-academic-codes/chocolate-lvsde/DataAbstraction"

type DataEmbeddingTechnique interface {
	EmbedData(dataAbstractionSet DataAbstraction.DataAbstractionSet)
}
