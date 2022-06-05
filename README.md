# Chocolate LVSDE

This project is an implementation of LVSDE dimensionality reduction technique written in Go programming language (majority of the code volume) and Python programming language, sometimes interoperating through C programming language interface of Go as the intermediate language interface, in addition to some codes written in Javascript, CSS and HTML.

Farshad Barahimi is the only code author for this project including but not limited to this file. All codes in this project including but not limited to this file are written by Farshad Barahimi.

Research contributors for the dimensionality reduction technique implemented: Farshad Barahimi and a researcher who is no longer doing research with Farshad Barahimi.

Note: This file only specifies the endorsement of Farshad Barahimi and no other endorsement should be implied from it.

LVSDE stands for Layered Vertex Splitting Data Embedding.

For more information about LVSDE dimensionality reduction technique (algorithm) look at the following arXiv preprint:

Farshad Barahimi, "Multi-point dimensionality reduction to improve projection layout reliability",  arXiv:2101.06224v3, 2022.

In short, LVSDE dimensionality reduction technique uses force-directed graph drawing on a neighbourhood graph through four phases and a number of iterations and splits the set of projected points into two possibly empty layers red and gray, and some of the data instances (data abstraction units) can have two projections in visual space.

The force-directed graph drawing algorithm used is based upon but modified from the Fruchterman and Reingold force-directed graph drawing algorithm.

In phase 1 a general layout of the neighbourhood graph is shaped.

In phase 2 some of the projected points are moved to gray layer and marked as ineffective (not applying forces and not forces being applied to them)

In phase 3, the ineffective mark is removed from projected points in gray layer but the projected points in red layer are marked as frozen (applying forces but not forces being applied to them)

In phase 4, some or all of the projected point in the gray layer are duplicated (being projection of the same data instance) as if the vertex in the neighbourhood graph is split into two vertices.

There are three preliminary steps, first one which is UMAP to thirty dimensions is optional. The second one is a distance transformation named neighbourhood-normalized distance transformation and the third one is building a neighbourhood graph.

The github repository for this project is designated at https://github.com/farshad-barahimi-academic-codes/chocolate-lvsde

Copyright notice for this code (this implementation of LVSDE):

Copyright (c) 2022 Farshad Barahimi. Licensed under the MIT license.

"Chocolate LVSDE" is the name of this implementation of LVSDE.

The purpose of writing this code is academic.

While this project sometimes uses UMAP and for comparison t-SNE, the author is not the author of UMAP or t-SNE and the copyright notice above does not correspond to implementations of UMAP and t-SNE in umap-learn library or scikit-learn library.