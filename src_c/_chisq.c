// Copyright (c) 2015-2021 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "ind.h"
#include "stats.h"


PyDoc_STRVAR(residuals__doc__,
"Calculate the residuals between a dataset and a model            \n\
                                                                  \n\
Parameters                                                        \n\
----------                                                        \n\
model: 1D ndarray                                                 \n\
    Model fit of data.                                            \n\
data: 1D ndarray                                                  \n\
    Data set array fitted by model.                               \n\
uncert: 1D ndarray                                                \n\
    Data uncertainties.                                           \n\
prioroff: 1D ndarray                                              \n\
    Parameter - prior offset                                      \n\
priorlow: 1D ndarray                                              \n\
    Prior lower uncertainty                                       \n\
priorup: 1D ndarray                                               \n\
    Prior upper uncertainty                                       \n\
                                                                  \n\
Returns                                                           \n\
-------                                                           \n\
residuals: 1D ndarray                                             \n\
   Residuals array.");

static PyObject *residuals(PyObject *self, PyObject *args){
  PyArrayObject *model,    /* Model of data                        */
                *data,     /* Data array                           */
                *uncert,   /* Data uncertainties                   */
                *prioroff=NULL, /* Parameter-prior offset          */
                *priorlow=NULL, /* Lower prior uncertainty         */
                *priorup=NULL,  /* Upper prior uncertainty         */
                *residuals;     /* Data--Model residuals           */
  int i, dsize, psize=0;
  npy_intp size[1]; /* Size of output numpy array                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "OOO|OOO", &model, &data, &uncert,
                                        &prioroff, &priorlow, &priorup)){
      return NULL;
  }
  /* Get data and prior arrays size:                               */
  dsize = (int)PyArray_DIM(model,    0);
  if (prioroff)
      psize = (int)PyArray_DIM(prioroff,0);
  size[0] = dsize + psize;

  /* Initialize resuduals array:                                   */
  residuals = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

  /* Calculate fit residuals:                                      */
  for(i=0; i<dsize; i++){
      INDd(residuals,i) = (INDd(model,i) - INDd(data,i))/INDd(uncert,i);
  }
  if (!prioroff){
      return PyArray_Return(residuals);
  }

  /* Add priors contribution:                                      */
  for(i=0; i<psize; i++){
      if (INDd(prioroff,i) > 0){
          INDd(residuals,(dsize+i)) = INDd(prioroff,i)/INDd(priorup, i);
      }else{
          INDd(residuals,(dsize+i)) = INDd(prioroff,i)/INDd(priorlow,i);
      }
  }
  return PyArray_Return(residuals);
}


PyDoc_STRVAR(chisq__doc__,
"Calculate chi-squared of a model fit to a data set:                 \n\
  chisq = sum{data points} ((data[i] -model[i])/error[i])**2.0   +   \n\
          sum{priors}      ((param[j]-prior[j])/prioruncert[j])**2.0 \n\
                                                                     \n\
Parameters:                                                          \n\
-----------                                                          \n\
model: 1D ndarray                                                    \n\
   Model fit of data.                                                \n\
data: 1D ndarray                                                     \n\
   Data set array fitted by model.                                   \n\
uncert: 1D ndarray                                                   \n\
   Data uncertainties.                                               \n\
prioroff: 1D ndarray                                                 \n\
   Parameter - prior offset                                          \n\
priorlow: 1D ndarray                                                 \n\
   Prior lower uncertainty                                           \n\
priorup: 1D ndarray                                                  \n\
   Prior upper uncertainty                                           \n\
                                                                     \n\
Returns:                                                             \n\
--------                                                             \n\
chisq: Float                                                         \n\
   The chi-squared value.                                            \n\
                                                                     \n\
Previous (uncredited) developers                                     \n\
--------------------------------                                     \n\
Nate Lust (UCF)");

static PyObject *chisq(PyObject *self, PyObject *args){
  PyArrayObject *model,         /* Model of data                   */
                *data,          /* Data array                      */
                *uncert,        /* Data uncertainties              */
                *prioroff=NULL, /* Parameter-prior offset          */
                *priorlow=NULL, /* Lower prior uncertainty         */
                *priorup =NULL; /* Upper prior uncertainty         */
  int dsize,        /* Array sizes                                 */
      i;            /* Auxilliary for-loop index                   */
  double chisq=0;   /* Chi-square                                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "OOO|OOO", &model, &data, &uncert,
                                        &prioroff, &priorlow, &priorup)){
      return NULL;
  }
  /* Get data and prior arrays size:                               */
  dsize = (int)PyArray_DIM(model, 0);

  /* Calculate model chi-squared:                                  */
  for(i=0; i<dsize; i++){
      chisq += pow((INDd(model,i)-INDd(data,i))/INDd(uncert,i), 2);
  }

  /* Calculate priors contribution:                                */
  if (prioroff != NULL)
      chisq += priors(prioroff, priorlow, priorup);

  return Py_BuildValue("d", chisq);
}


PyDoc_STRVAR(_chisqmod__doc__, "Residuals and chi-squared calculation.");

static PyMethodDef _chisq_methods[] = {
    {"chisq",     chisq,     METH_VARARGS, chisq__doc__},
    {"residuals", residuals, METH_VARARGS, residuals__doc__},
    {NULL,        NULL,      0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_chisq",
    _chisqmod__doc__,
    -1,
    _chisq_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__chisq (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_chisq(void){
  Py_InitModule3("_chisq", _chisq_methods, _chisqmod__doc__);
  import_array();
}
#endif
