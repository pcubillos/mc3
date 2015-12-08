// ******************************* START LICENSE *****************************
//
// Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
// model-parameter best-fitting values and Bayesian posterior
// distributions.
//
// This project was completed with the support of the NASA Planetary
// Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
// Joseph Harrington.  Principal developers included graduate student
// Patricio E. Cubillos and programmer Madison Stemm.  Statistical advice
// came from Thomas J. Loredo and Nate B. Lust.
//
// Copyright (C) 2015 University of Central Florida.  All rights reserved.
//
// This is a test version only, and may not be redistributed to any third
// party.  Please refer such requests to us.  This program is distributed
// in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.
//
// Our intent is to release this software under an open-source,
// reproducible-research license, once the code is mature and the first
// research paper describing the code has been accepted for publication
// in a peer-reviewed journal.  We are committed to development in the
// open, and have posted this code on github.com so that others can test
// it and give us feedback.  However, until its first publication and
// first stable release, we do not permit others to redistribute the code
// in either original or modified form, nor to publish work based in
// whole or in part on the output of this code.  By downloading, running,
// or modifying this code, you agree to these conditions.  We do
// encourage sharing any modifications with us and discussing them
// openly.
//
// We welcome your feedback, but do not guarantee support.  Please send
// feedback or inquiries to:
//
// Joseph Harrington <jh@physics.ucf.edu>
// Patricio Cubillos <pcubillos@fulbrightmail.org>
//
// or alternatively,
//
// Joseph Harrington and Patricio Cubillos
// UCF PSB 441
// 4111 Libra Drive
// Orlando, FL 32816-2385
// USA
//
// Thank you for using MC3!
// ******************************* END LICENSE *******************************

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "ind.h"
#include "stats.h"

PyDoc_STRVAR(residuals__doc__,
"Calculate the residuals between a dataset and a model            \n\
                                                                  \n\
Parameters:                                                       \n\
-----------                                                       \n\
model: 1D ndarray                                                 \n\
   Model fit of data.                                             \n\
data: 1D ndarray                                                  \n\
   Data set array fitted by model.                                \n\
errors: 1D ndarray                                                \n\
   Data uncertainties.                                            \n\
prioroff: 1D ndarray                                              \n\
   Parameter - prior offset                                       \n\
priorlow: 1D ndarray                                              \n\
   Prior lower uncertainty                                        \n\
priorup: 1D ndarray                                               \n\
   Prior upper uncertainty                                        \n\
                                                                  \n\
Returns:                                                          \n\
--------                                                          \n\
residuals: 1D ndarray                                             \n\
   Residuals array.                                               \n\
                                                                  \n\
Modification History:                                             \n\
---------------------                                             \n\
2011-01-08  patricio  Initial version, Based on chisq function");

static PyObject *residuals(PyObject *self, PyObject *args){
  PyArrayObject *model,    /* Model of data                        */
                *data,     /* Data array                           */
                *errors,   /* Data uncertainties                   */
                *prioroff, /* Parameter-prior offset               */
                *priorlow, /* Lower prior uncertainty              */
                *priorup,  /* Upper prior uncertainty              */
                *residuals;/* Data--Model residuals                */
  int dsize, psize, /* Array sizes                                 */
      i;            /* Auxilliary for-loop index                   */
  npy_intp size[1]; /* Size of output numpy array                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "OOOOOO", &model, &data, &errors,
                                       &prioroff, &priorlow, &priorup)){
    return NULL;
  }
  /* Get data and prior arrays size:                               */
  dsize = (int)PyArray_DIM(model,    0);
  psize = (int)PyArray_DIM(prioroff, 0);
  size[0] = dsize + psize;

  /* Initialize resuduals array:                                   */
  residuals = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

  /* Calculate fit residuals:                                      */
  for(i=0; i<dsize; i++){
    INDd(residuals,i) = (INDd(model,i) - INDd(data,i))/INDd(errors,i);
  }
  /* Calculate priors contribution:                                */
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
errors: 1D ndarray                                                   \n\
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
njchisq: Float                                                       \n\
   No-Jeffrey's chi-squared                                          \n\
                                                                     \n\
Modification History:                                                \n\
---------------------                                                \n\
2011-01-08  Nate      Initial version, Nate Lust, UCF                \n\
                      natelust at linux dot com                      \n\
2014-05-09  Patricio  Added priors support.  pcubillos@fulbrightmail.org");

static PyObject *chisq(PyObject *self, PyObject *args){
  PyArrayObject *model,         /* Model of data                   */
                *data,          /* Data array                      */
                *errors,        /* Data uncertainties              */
                *prioroff=NULL, /* Parameter-prior offset          */
                *priorlow=NULL, /* Lower prior uncertainty         */
                *priorup =NULL; /* Upper prior uncertainty         */
  int dsize,        /* Array sizes                                 */
      i;            /* Auxilliary for-loop index                   */
  double chisq=0;   /* Chi-square                                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "OOO|OOO", &model, &data, &errors,
                                        &prioroff, &priorlow, &priorup)){
    return NULL;
  }
  /* Get data and prior arrays size:                               */
  dsize = (int)PyArray_DIM(model, 0);

  /* Calculate model chi-squared:                                  */
  for(i=0; i<dsize; i++){
    chisq += pow((INDd(model,i)-INDd(data,i))/INDd(errors,i), 2);
  }

  /* Calculate priors contribution:                                */
  if (prioroff != NULL)
    chisq += priors(prioroff, priorlow, priorup);

  return Py_BuildValue("d", chisq);
}


PyDoc_STRVAR(chisqmod__doc__, "Residuals and chi-squared calculation.");

static PyMethodDef chisq_methods[] = {
        {"chisq",     chisq,     METH_VARARGS, chisq__doc__},
        {"residuals", residuals, METH_VARARGS, residuals__doc__},
        {NULL,        NULL,      0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "chisq",
    chisqmod__doc__,
    -1,
    chisq_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_chisq (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initchisq(void){
  Py_InitModule3("chisq", chisq_methods, chisqmod__doc__);
  import_array();
}
#endif
