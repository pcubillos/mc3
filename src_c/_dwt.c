// Copyright (c) 2015-2021 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "ind.h"
#include "wavelet.h"
#include "stats.h"


PyDoc_STRVAR(chisq__doc__,
"Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared) \n\
based on Carter & Winn (2009), ApJ 704, 51.                           \n\
                                                                      \n\
Parameters                                                            \n\
----------                                                            \n\
params: 1D float ndarray                                              \n\
    Model parameters (including the tree noise parameters: gamma,     \n\
    sigma_r, sigma_w; which must be the last three elements in params).\n\
model: 1D ndarray                                                     \n\
    Model fit of data.                                                \n\
data: 1D ndarray                                                      \n\
    Data set array fitted by model.                                   \n\
prioroff: 1D ndarray                                                  \n\
    Parameter - prior offset                                          \n\
priorlow: 1D ndarray                                                  \n\
    Prior lower uncertainty                                           \n\
priorup: 1D ndarray                                                   \n\
    Prior upper uncertainty                                           \n\
                                                                      \n\
Returns:                                                              \n\
--------                                                              \n\
chisq: Float                                                          \n\
    Wavelet-based (pseudo) chi-squared.                               \n\
                                                                      \n\
Notes                                                                 \n\
-----                                                                 \n\
- If the residuals array size is not of the form 2**N, the routine    \n\
zero-padds the array until this condition is satisfied.               \n\
- The current code only supports gamma=1.                             \n\
                                                                      \n\
Examples                                                              \n\
--------                                                              \n\
>>> import _dwt as dwt                                                \n\
>>> import numpy as np                                                \n\
                                                                      \n\
>>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])       \n\
>>> model = np.ones(8)                                                \n\
>>> pars = np.array([1.0, 0.1, 0.1])                                  \n\
>>> chisq = dwt.chisq(params, model, data)                            \n\
>>> print(chisq)                                                      \n\
1693.22308882");

static PyObject *chisq(PyObject *self, PyObject *args){
  PyArrayObject *params, *data, *model, *prioroff=NULL,
                *priorlow=NULL, *priorup=NULL; /* Inputs */
  double gamma, sigmar, sigmaw, res2m,
         sW2, sS2,   /* Variance of wavelet and scaling coeffs        */
         chisq,      /* Wavelet-based chi-squared                     */
         *wres; /* Extended residuals array                           */
  int rsize,   /* Input residuals-array size                          */
      wrsize,  /* Extended residuals-array size (as 2^M)              */
      npars,
      M,       /* Number of DWT scales                                */
      n, j, m; /* Auxilliary for-loop indices                         */
  double g = 0.72134752; /* g-factor of the covariance of the wavelet */
                         /* coefficients: g(gamma=1) = 1.0/(2*ln(2))  */
  /* Load inputs:                                                     */
  if (!PyArg_ParseTuple(args, "OOO|OOO", &params, &model, &data,
                                        &prioroff, &priorlow, &priorup))
      return NULL;

  /* Unpack parameters array:                                         */
  npars = (int)PyArray_DIM(params, 0);
  gamma  = INDd(params, (npars-3));
  sigmar = INDd(params, (npars-2));
  sigmaw = INDd(params, (npars-1));

  /* Get data array size:                                             */
  rsize = (int)PyArray_DIM(data, 0);  /* Get residuals vector size    */
  M = ceil(1.0*log2(rsize));     /* Number of scales                  */

  /* Expand res to a size proportional to 2^M (zero padding)          */
  wrsize = (int)pow(2, M);     /* Expanded size                       */
  wres = (double *)malloc(wrsize *sizeof(double));
  for(j=0; j<rsize; j++)
      wres[j] = INDd(data, j) - INDd(model, j);
  for(j=rsize; j<wrsize; j++) /* Zero-pad the extended values         */
      wres[j] = 0.0;

  /* Calculate the DWT of the residuals:                              */
  dwt(wres, rsize, 1);

  /* Equation (34) of CW2009, square of sigma_S:                      */
  sS2 = sigmar*sigmar*pow(2.0,-gamma)*g + sigmaw*sigmaw;
  /* Second term in right-hand side of equation 32:                   */
  chisq = wres[0]*wres[0]/sS2 + wres[1]*wres[1]/sS2 + 2.0*log(2*M_PI*sS2);

  for (m=1; m<M; m++){  /* Number of scales                           */
      /* Equation (33) of CW2009, sigma_W squared:                    */
      sW2 = sigmar*sigmar*pow(2.0,-gamma*m) + sigmaw*sigmaw;
      n = pow(2, m);      /* Number of coefficients per scale         */
      res2m = 0.0;        /* Sum of residuals squared at scale m      */
      for (j=0; j<n; j++){
          res2m += wres[n+j]*wres[n+j];
      }
      chisq += res2m/sW2 + n*log(2*M_PI*sW2);
  }

  /* Add priors contribution:                                         */
  if (prioroff != NULL)
      chisq += priors(prioroff, priorlow, priorup);

  /* Free the allocated arrays and return chi-squared:                */
  free(wres);
  return Py_BuildValue("d", chisq);
}


PyDoc_STRVAR(daub4__doc__,
"1D discrete wavelet transform using the Daubechies 4-parameter wavelet\n\
                                                                       \n\
Parameters                                                             \n\
----------                                                             \n\
array: 1D ndarray                                                      \n\
    Data array to which to apply the DWT.                              \n\
inverse: bool                                                          \n\
   If False, calculate the DWT,                                        \n\
   If True, calculate the inverse DWT.                                 \n\
                                                                       \n\
Notes                                                                  \n\
-----                                                                  \n\
The input vector must have length 2**M with M an integer, otherwise the\n\
output will zero-padded to the next size of the form 2**M.             \n\
                                                                       \n\
Examples                                                               \n\
--------                                                               \n\
>>> import numpy as np                                                 \n\
>>> improt matplotlib.pyplot as plt                                    \n\
>>> import mc3.stats as ms                                             \n\
                                                                       \n\
>>> # Calculate the inverse DWT for a unit vector:                     \n\
>>> nx = 1024                                                          \n\
>>> e4 = np.zeros(nx)                                                  \n\
>>> e4[4] = 1.0                                                        \n\
>>> ie4 = ms.dwt_daub4(e4, True)                                       \n\
>>> # Plot the inverse DWT                                             \n\
>>> plt.figure(0)                                                      \n\
>>> plt.clf()                                                          \n\
>>> plt.plot(np.arange(nx), ie4");

static PyObject *daub4(PyObject *self, PyObject *args){
    PyArrayObject *array, *dwt_array;
      double *ptr_array;
      int isign,  /* Sign indicator to perform DWT or inverse DWT */
          asize, dwt_size,
          M, j;

      if (!PyArg_ParseTuple(args, "Oi", &array, &isign))
          return NULL;

      /* Expand to a size proportional to 2^M (zero padding) */
      asize = (int)PyArray_DIM(array, 0);
      M = ceil(1.0*log2(asize));
      dwt_size = (int)pow(2, M);

      /* Allocate memory for pointer with the data: */
      ptr_array = malloc(dwt_size*sizeof(double));
      for(j=0; j<asize; j++)
          ptr_array[j] = INDd(array, j);
      for(j=asize; j<dwt_size; j++)
          ptr_array[j] = 0.0;

      /* Calculate the discrete wavelet transform: */
      dwt(ptr_array, dwt_size, isign);

      dwt_array = (PyArrayObject *) PyArray_FromDims(1, &dwt_size, NPY_DOUBLE);
      for(j=0; j<dwt_size; j++)
          INDd(dwt_array,j) = ptr_array[j];

      free(ptr_array);
      return Py_BuildValue("O", dwt_array);
}


/* The module doc string    */
PyDoc_STRVAR(_dwt__doc__, "Discrete Wavelet Transform wrapper for Python.");

/* A list of all the methods defined by this module. */
static PyMethodDef _dwt_methods[] = {
    {"chisq", chisq, METH_VARARGS, chisq__doc__},
    {"daub4", daub4, METH_VARARGS, daub4__doc__},
    {NULL,    NULL,  0,            NULL}    /* sentinel */
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_dwt",
    _dwt__doc__,
    -1,
    _dwt_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit__dwt (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void init_dwt(void){
  Py_InitModule3("_dwt", _dwt_methods, _dwt__doc__);
  import_array();
}
#endif
