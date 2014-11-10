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
// Copyright (C) 2014 University of Central Florida.  All rights reserved.
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
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include "stats.h"

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))


PyDoc_STRVAR(weightedbin__doc__,
"Calculate the weighted mean (for known bin standard deviation)   \n\
                                                                  \n\
  Parameters:                                                     \n\
  -----------                                                     \n\
  data: 1D ndarray                                                \n\
    A time-series dataset.                                        \n\
  binsize: Integer                                                \n\
    Number of data points per bin.                                \n\
  uncert: 1D ndarray                                              \n\
    Uncertainties of data.                                        \n\
  std: 1D ndarray                                                 \n\
    Standard deviation of the bins (for the given uncert).        \n\
                                                                  \n\
  Notes:                                                          \n\
  ------                                                          \n\
  If uncert and std are not provided, use flat weights.           \n\
                                                                  \n\
  Returns:                                                        \n\
  --------                                                        \n\
  bindat: 1D ndarray                                              \n\
     Mean-weighted binned data (using 1/uncert**2 as weights).    \n\
                                                                  \n\
  Modification History:                                           \n\
  ---------------------                                           \n\
  2014-05-15  patricio  Documented, implemented in C.             \n\
                        pcubillos@fulbrightmail.org");

static PyObject *weightedbin(PyObject *self, PyObject *args){
  PyArrayObject *data,        /* Data array                        */
                *uncert=NULL, /* Data uncertainties                */
                *std=NULL,    /* Standard deviation of bins        */
                *bindat;      /* Mean-weighted binned data         */
  int dsize, binsize,         /* Data and bin sizes                */
      nbins,                  /* Number of bins                    */
      i, j, idx;              /* Auxilliary for-loop indices       */
  npy_intp size[1];           /* Size of binned arrays             */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "Oi|OO", &data, &binsize, &uncert, &std))
    return NULL;

  /* Get data array size:                                          */
  dsize = data->dimensions[0];
  nbins = dsize/binsize;

  /* Initialize numpy array:                                       */
  size[0] = nbins;
  bindat   = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);

  for (i=0; i<nbins; i++){
    IND(bindat,i) = 0.0;
    for (j=0; j<binsize; j++){
      idx = i*binsize+j;
      if (!uncert){  /* Flat-weights mean                           */
        IND(bindat,i) += IND(data, idx) / binsize;
      }else{          /* Weighted mean                               */
        IND(bindat,i) += IND(std,i) * IND(data,  idx)    /
                                  pow(IND(uncert,idx), 2);
      }
    }
  }
  Py_XDECREF(size);
  return Py_BuildValue("N", bindat);
}


PyDoc_STRVAR(binarray__doc__,
"Compute the binned root-mean-square and extrapolated             \n\
 Gaussian-noise rms for a dataset.                                \n\
                                                                  \n\
  Parameters:                                                     \n\
  -----------                                                     \n\
  data: 1D ndarray                                                \n\
    A time-series dataset.                                        \n\
  uncert: 1D ndarray                                              \n\
    Uncertainties of data.                                        \n\
  indp: 1D ndarray                                                \n\
    Independent variable.                                         \n\
  binsize: Integer                                                \n\
    Number of data points per bin.                                \n\
                                                                  \n\
  Returns:                                                        \n\
  --------                                                        \n\
  bindata: 1D ndarray                                             \n\
     Mean-weighted binned data (using 1/unc**2 as weights).       \n\
  binunc: 1D ndarray                                              \n\
     Standard deviation of the binned data points.                \n\
  binindp: 1D ndarray                                             \n\
     Mean-averaged binned indp.                                   \n\
                                                                  \n\
  Modification History:                                           \n\
  ---------------------                                           \n\
  2012-       kevin     Initial python implementation by          \n\
                        Kevin Stevenson, UCF.                     \n\
  2012-01-21  matt      Added integer conversion by Matt Hardin.  \n\
  2014-05-15  patricio  Documented, implemented in C.             \n\
                        pcubillos@fulbrightmail.org");

static PyObject *binarray(PyObject *self, PyObject *args){
  PyArrayObject *data,     /* Data array                           */
                *uncert,   /* Data uncertainties                   */
                *indp,     /* Independdent variable                */
                *bindat,   /* Binned data array                    */
                *binunc,   /* Binned uncertainty                   */
                *binindp;  /* Binned indp                          */
  int dsize,      /* Size of data array                            */
      nbins,      /* Size of binned arrays                         */
      binsize,    /* Number of points per bin                      */
      i;          /* Auxilliary for-loop index                     */
  double *pdata, *punc, *pindp; /* Pointers to data arrays         */
  npy_intp size[1]; /* Size of binned arrays                       */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "OOOi", &data, &uncert, &indp, &binsize)){
    return NULL;
  }
  /* Get data array size:                                          */
  dsize = data->dimensions[0];
  /* Set the number of bins:                                       */
  nbins = dsize/binsize;
  size[0] = nbins;

  /* Initialize numpy arrays:                                      */
  bindat   = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);
  binunc   = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);
  binindp  = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);

  /* Initialize pointers:                                          */
  pdata = (double *)malloc(dsize*sizeof(double));
  punc  = (double *)malloc(dsize*sizeof(double));
  pindp = (double *)malloc(dsize*sizeof(double));
  for (i=0; i<dsize; i++){
    pdata[i] = IND(data,  i);
    punc [i] = IND(uncert,i);
    pindp[i] = IND(indp,  i);
  }

  /* Calculate the binned data:                                    */
  bindata(pdata, punc, pindp, dsize, binsize, bindat, binunc, binindp);

  /* Free arrays and return:                                       */
  free(pdata);
  free(punc);
  free(pindp);
  Py_XDECREF(size);
  return Py_BuildValue("[N,N,N]", bindat, binunc, binindp);
}


PyDoc_STRVAR(binarraymod__doc__, "Weighted mean binning");

static PyMethodDef binarray_methods[] = {
        {"weightedbin",  weightedbin, METH_VARARGS, weightedbin__doc__},
        {"binarray",     binarray,    METH_VARARGS, binarray__doc__},
        {NULL,           NULL,        0,            NULL}
};

void initbinarray(void){
  Py_InitModule3("binarray", binarray_methods, binarraymod__doc__);
  import_array();
}

