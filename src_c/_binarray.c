// Copyright (c) 2015-2021 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "ind.h"
#include "stats.h"


PyDoc_STRVAR(binarray__doc__,
"Compute the weighted-mean binned values and standard deviation of\n\
an array using 1/unc**2 as weights.                             \n\
                                                                \n\
Parameters                                                      \n\
----------                                                      \n\
data: 1D ndarray                                                \n\
    A time-series dataset.                                      \n\
binsize: Integer                                                \n\
    Number of data points per bin.                              \n\
uncert: 1D ndarray                                              \n\
    Uncertainties of data.                                      \n\
                                                                \n\
Returns                                                         \n\
-------                                                         \n\
bindata: 1D ndarray                                             \n\
    Mean-weighted binned data.                                  \n\
binstd: 1D ndarray                                              \n\
    Standard deviation of the binned data points.");

static PyObject *binarray(PyObject *self, PyObject *args){
    PyArrayObject *data,
                  *uncert=NULL,
                  *bindat,
                  *binstd;
    int dsize, nbins, binsize, i, j;
    npy_intp size[1];

    /* Unpack arguments:                                             */
    if(!PyArg_ParseTuple(args, "Oi|O", &data, &binsize, &uncert)){
        return NULL;
    }
    /* Get data array size:                                          */
    dsize = (int)PyArray_DIM(data, 0);
    /* Set the number of bins:                                       */
    size[0] = nbins = dsize/binsize;

    /* Initialize numpy arrays:                                      */
    bindat = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    /* Un-weighted means: */
    if (!uncert){
        for (i=0; i<nbins; i++){
            INDd(bindat,i) = 0.0;
            for (j=i*binsize; j<(i+1)*binsize; j++)
                INDd(bindat,i) += INDd(data,j);
            INDd(bindat,i) /= binsize;
        }
        Py_XDECREF(size);
        return Py_BuildValue("N", bindat);
    }

    /* Weighted means: */
    binstd = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    for (i=0; i<nbins; i++){
        INDd(binstd,i) = 0.0;
        INDd(bindat,i) = 0.0;
        for (j=i*binsize; j<(i+1)*binsize; j++){
            INDd(binstd,i) += 1.0 / (INDd(uncert,j)*INDd(uncert,j));
            INDd(bindat,i) += INDd(data,j) / (INDd(uncert,j)*INDd(uncert,j));
        }
        INDd(binstd,i) = sqrt(1.0/INDd(binstd,i));
        INDd(bindat,i) *= INDd(binstd,i)*INDd(binstd,i);
    }
    Py_XDECREF(size);
    return Py_BuildValue("[N,N]", bindat, binstd);
}


PyDoc_STRVAR(_binarraymod__doc__, "Weighted mean binning");

static PyMethodDef _binarray_methods[] = {
    {"binarray",     binarray,    METH_VARARGS, binarray__doc__},
    {NULL,           NULL,        0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_binarray",
    _binarraymod__doc__,
    -1,
    _binarray_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__binarray (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_binarray(void){
    Py_InitModule3("_binarray", _binarray_methods, _binarraymod__doc__);
    import_array();
}
#endif

