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

package PythonInterop

// #cgo CFLAGS: -IC:/Users/TheUser/miniconda3/envs/temp-env/include -DMS_WIN64
// #cgo LDFLAGS: -LC:/Users/TheUser/miniconda3/envs/temp-env/libs -lpython310 -lpthread -lm
// #include <Python.h>
import "C"
import "unsafe"

func RunPythonFunction(functionCode string, functionName string, functionParameter []float64, functionOutputSize int) []float64 {
	functionParameterSize := len(functionParameter)

	defer C.Py_Finalize()
	C.Py_Initialize()

	moduleName := C.CString("functionModule")
	defer C.free(unsafe.Pointer(moduleName))

	moduleObject := C.PyModule_New(moduleName)
	defer C.Py_DecRef(moduleObject)

	temp1 := C.CString("__file__")
	defer C.free(unsafe.Pointer(temp1))
	temp2 := C.CString("")
	defer C.free(unsafe.Pointer(temp2))
	C.PyModule_AddStringConstant(moduleObject, temp1, temp2)

	moduleDictionaryObject := C.PyModule_GetDict(moduleObject)
	defer C.Py_DecRef(moduleDictionaryObject)

	emptyDictionary := C.PyDict_New()
	defer C.Py_DecRef(emptyDictionary)

	functionCodeC := C.CString(functionCode)
	defer C.free(unsafe.Pointer(functionCodeC))
	functionCodeObject := C.PyRun_String(functionCodeC, C.Py_file_input, emptyDictionary, moduleDictionaryObject)
	defer C.Py_DecRef(functionCodeObject)

	functionNameC := C.CString(functionName)
	defer C.free(unsafe.Pointer(functionNameC))
	functionObject := C.PyObject_GetAttrString(moduleObject, functionNameC)
	defer C.Py_DecRef(functionObject)

	functionParameterObject := C.PyTuple_New(C.longlong(functionParameterSize))
	defer C.Py_DecRef(functionParameterObject)

	for i := 0; i < functionParameterSize; i++ {
		functionParameterTempObject := C.PyFloat_FromDouble(C.double(functionParameter[i]))
		C.PyTuple_SetItem(functionParameterObject, C.longlong(i), functionParameterTempObject)
	}

	functionReturnObject := C.PyObject_CallObject(functionObject, functionParameterObject)
	defer C.Py_DecRef(functionReturnObject)

	if functionReturnObject == nil {
		C.PyErr_PrintEx(0)
		C.PyErr_Clear()
	}

	functionOutput := make([]float64, functionOutputSize)

	for i := 0; i < functionOutputSize; i++ {
		functionOutputTempObject := C.PyTuple_GetItem(functionReturnObject, C.longlong(i))
		functionOutput[i] = float64(C.PyFloat_AsDouble(functionOutputTempObject))

	}

	return functionOutput
}
