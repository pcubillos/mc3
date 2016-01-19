// Copyright (c) 2015-2016 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>

#include "ind.h"
#include "wavelet.h"
#include "stats.h"

PyDoc_STRVAR(wlikelihood__doc__,
"Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared) \n\
                                                                      \n\
Parameters:                                                           \n\
-----------                                                           \n\
params: 1D ndarray                                                    \n\
   Noise parameter estimators: (gamma, sigma_r, sigma_w)              \n\
res:    1D ndarray                                                    \n\
   (data - model) array of residuals                                  \n\
prioroff: 1D ndarray                                                  \n\
   Parameter - prior offset                                           \n\
priorlow: 1D ndarray                                                  \n\
   Prior lower uncertainty                                            \n\
priorup: 1D ndarray                                                   \n\
   Prior upper uncertainty                                            \n\
                                                                      \n\
Returns:                                                              \n\
--------                                                              \n\
chisq: Float                                                          \n\
   Wavelet-based (pseudo) chi-squared.                                \n\
njchisq: Float                                                        \n\
   No-Jeffrey's chi-square                                            \n\
                                                                      \n\
Notes:                                                                \n\
------                                                                \n\
- Implementation based on Carter & Winn (2009), ApJ 704, 51.          \n\
- If the residuals array size is not of the form 2**N, the routine    \n\
zero-padds the array until this condition is satisfied.               \n\
- The current code only supports gamma=1.                             \n\
                                                                      \n\
Example:                                                              \n\
--------                                                              \n\
>>>import dwt as dwt                                                  \n\
>>>import numpy as np                                                 \n\
                                                                      \n\
>>>x = np.array([1.0, -1, 2, -3, -2, 1, 1, -1])                       \n\
>>>pars = np.array([1.0, 0.1, 0.1])                                   \n\
>>>chisq = dwt.wlikelihood(pars,x)                                    \n\
>>>print(chisq)                                                       \n\
1693.22308882");

static PyObject *wlikelihood(PyObject *self, PyObject *args){
  PyArrayObject *params, *res, *prioroff=NULL,
                *priorlow=NULL, *priorup=NULL; /* Inputs */
  double gamma, sigmar, sigmaw, res2m,
         sW2, sS2,   /* Variance of wavelet and scaling coeffs        */
         chisq,      /* Wavelet-based chi-squared                     */
         *jchisq, jc,   /* Jeffrey's chi-squared                      */
         *wres; /* Extended residuals array                           */
  int rsize,   /* Input residuals-array size                          */
      wrsize,  /* Extended residuals-array size (as 2^M)              */
      M,       /* Number of DWT scales                                */
      n, j, m; /* Auxilliary for-loop indices                         */
  double g = 0.72134752; /* g-factor of the covariance of the wavelet */
                         /* coefficients: g(gamma=1) = 1.0/(2*ln(2))  */
  /* Load inputs:                                                     */
  if (!PyArg_ParseTuple(args, "OO|OOO", &params, &res,
                                        &prioroff, &priorlow, &priorup))
    return NULL;

  /* Unpack parameters array:                                         */
  gamma  = INDd(params, 0);
  sigmar = INDd(params, 1);
  sigmaw = INDd(params, 2);

  /* Get data array size:                                             */
  rsize = (int)PyArray_DIM(res, 0);   /* Get residuals vector size    */
  M = ceil(1.0*log2(rsize));          /* Number of scales             */

  /* Expand res to a size proportional to 2^M (zero padding)          */
  wrsize = (int)pow(2, M);     /* Expanded size                       */
  wres = (double *)malloc(wrsize *sizeof(double));
  for(j=0; j<rsize; j++)
    wres[j] = INDd(res, j);
  for(j=rsize; j<wrsize; j++) /* Zero-pad the extended values         */
    wres[j] = 0.0;

  /* Calculate the DWT of the residuals:                              */
  dwt(wres, rsize, 1);

  /* Equation 34 of CW2009, square of sigma_S:                        */
  sS2 = sigmar*sigmar*pow(2.0,-gamma)*g + sigmaw*sigmaw;
  /* Second term in right-hand side of equation 32:                   */
  chisq = wres[0]*wres[0]/sS2 + wres[1]*wres[1]/sS2 + 2.0*log(2*M_PI*sS2);

  for (m=1; m<M; m++){  /* Number of scales                           */
    /* Equation 33 of CW2009, sigma_W squared:                        */
    sW2 = sigmar*sigmar*pow(2.0,-gamma*m) + sigmaw*sigmaw;
    n = pow(2, m);      /* Number of coefficients per scale           */
    res2m = 0.0;        /* Sum of residuals squared at scale m        */
    for (j=0; j<n; j++){
      res2m += wres[n+j]*wres[n+j];
    }
    chisq += res2m/sW2 + n*log(2*M_PI*sW2);
  }

  /* Add priors contribution:                                         */
  jchisq = &jc;
  if (prioroff != NULL)
    chisq += priors(prioroff, priorlow, priorup, jchisq);

  /* Free the allocated arrays and return chi-squared:                */
  free(wres);
  return Py_BuildValue("[d,d]", chisq, chisq-jchisq[0]);
}


PyDoc_STRVAR(daubechies4__doc__,
"1D discrete wavelet transform using the Daubechies 4-parameter wavelet\n\
                                                                    \n\
Parameters:                                                         \n\
-----------                                                         \n\
vector: 1D ndarray                                                  \n\
   Data vector which to apply the DWT.                              \n\
isign: Integer                                                      \n\
   If isign= 1, calculate the DWT,                                  \n\
   If isign=-1, calculate the inverse DWT.                          \n\
                                                                    \n\
Notes:                                                              \n\
------                                                              \n\
The input vector must have length 2**M with M an integer.           \n\
                                                                    \n\
Example:                                                            \n\
--------                                                            \n\
>>>import numpy as np                                               \n\
>>>import matplotlib.pyplot as plt                                  \n\
>>>import dwt as dwt                                                \n\
                                                                    \n\
>>>nx = 1024  # Calculate the inverse DWT for a unit vector:        \n\
>>>e4 = np.zeros(nx)                                                \n\
>>>e4[4] = 1.0                                                      \n\
>>>ie4 = dwt.daubechies4(e4, -1)                                    \n\
                                                                    \n\
>>>plt.figure(0) # Plot the inverse DWT                             \n\
>>>plt.clf()                                                        \n\
>>>plt.plot(np.arange(nx), ie4)                                     \n\
>>>plt.xlim(0,nx)");

static PyObject *daubechies4(PyObject *self, PyObject *args){
  PyArrayObject *vector,     /* Input data vector                   */
                *dwtvector;  /* DWT of vector                       */
  double *ptrvector;   /* Pointer to data vector                    */
  int isign,  /* Sign indicator to perform DWT or inverse DWT       */
      vsize,  /* Size of data vector                                */
      j;      /* Auxilliary for-loop index                          */

  /* Load inputs:                                                   */
  if (!PyArg_ParseTuple(args, "Oi", &vector, &isign))
    return NULL;

  /* Get size of input vector:                                      */
  vsize = (int)PyArray_DIM(vector, 0);

  /* Allocate memory for pointer with the data:                     */
  ptrvector = malloc(vsize * sizeof(double));
  /* copy data into pointer:                                        */
  for(j=0; j<vsize; j++)
    ptrvector[j] = INDd(vector, j);

  /* Calculate the discrete wavelet transform:                      */
  dwt(ptrvector, vsize, isign);

  /* Restore values into a PyArrayObject:                           */
  dwtvector = (PyArrayObject *) PyArray_FromDims(1, &vsize, NPY_DOUBLE);
  for(j=0; j<vsize; j++)
    INDd(dwtvector, j) = ptrvector[j];

  /* Freee allocated arrays and return the DWT:                     */
  free(ptrvector);
  return Py_BuildValue("O", dwtvector);
}


/* The module doc string    */
PyDoc_STRVAR(dwt__doc__, "Discrete Wavelet Transform for python.");

/* A list of all the methods defined by this module. */
static PyMethodDef dwt_methods[] = {
    {"wlikelihood", wlikelihood, METH_VARARGS, wlikelihood__doc__},
    {"daubechies4", daubechies4, METH_VARARGS, daubechies4__doc__},
    {NULL,          NULL,        0,            NULL}    /* sentinel */
};


#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "dwt",
    dwt__doc__,
    -1,
    dwt_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_dwt (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initdwt(void){
  Py_InitModule3("dwt", dwt_methods, dwt__doc__);
  import_array();
}
#endif
