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


PyDoc_STRVAR(binrms__doc__,
"Compute the binned root-mean-square and extrapolated             \n\
Gaussian-noise rms for a dataset.                                 \n\
                                                                  \n\
  Parameters:                                                     \n\
  -----------                                                     \n\
  data: 1D ndarray                                                \n\
    A time-series dataset.                                        \n\
  maxbins: Scalar                                                 \n\
    Maximum bin size to calculate.                                \n\
  binstep: Integer                                                \n\
    Stepsize of binning indexing.                                 \n\
                                                                  \n\
  Returns:                                                        \n\
  --------                                                        \n\
  rms: 1D ndarray                                                 \n\
     RMS of binned data.                                          \n\
  rmserr: 1D ndarray                                              \n\
     RMS uncertainties.                                           \n\
  stderr: 1D ndarray                                              \n\
     Extrapolated RMS for Gaussian noise.                         \n\
  binsz: 1D ndarray                                               \n\
     Bin sizes.                                                   \n\
                                                                  \n\
  Modification History:                                           \n\
  ---------------------                                           \n\
  2012-       kevin     Initial python implementation by          \n\
                        Kevin Stevenson, UCF.                     \n\
  2012-01-21  matt      Added integer conversion by Matt Hardin.  \n\
  2014-05-15  patricio  Documented, implemented in C.             \n\
                        pcubillos@fulbrightmail.org");

static PyObject *binrms(PyObject *self, PyObject *args){
  PyArrayObject *data,     /* Data array                           */
                *datarms,  /* RMS of the binned data               */
                *rmserr,   /* RMS uncertainties                    */
                *gausserr, /* Gaussian-noise rms extrapolation     */
                *binsize;  /* Bin sizes                            */
  int dsize,      /* Data array size                               */
      maxbins=-1, /* Maximum bin size                              */
      binstep=1,  /* */
      i, j,       /* Auxilliary for-loop index                     */
      M;          /* Number of data bins for given bin size        */
  double *bindata, /* Binned data pointer                          */
         *arr,     /* Data array pointer                           */
         stddata;  /* Standard deviation of data                   */

  npy_intp size[1]; /* Size of output numpy array                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "O|ii", &data, &maxbins, &binstep)){
    return NULL;
  }
  /* Get data array size:                                          */
  dsize = data->dimensions[0];
  /* Set default maxbins:                                          */
  if (maxbins == -1)
    maxbins = dsize/2;

  /* Initialize numpy arrays:                                      */
  size[0] = (maxbins-1)/binstep + 1;
  datarms  = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);
  rmserr   = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);
  gausserr = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);
  binsize  = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);

  /* Initialize pointers:                                          */
  bindata = (double *)malloc(dsize*sizeof(double));
  arr     = (double *)malloc(dsize*sizeof(double));
  for (i=0; i<dsize; i++)
    arr[i] = IND(data,i);

  /* Calculate standard deviation of data:                         */
  stddata = std(arr, dsize);

  for(i=0; i<size[0]; i++){
    /* Set bin size and number of bins:                            */
    IND(binsize,i) = 1 + i*binstep;
    M = dsize/(int)IND(binsize,i);
    /* Bin the dataset:                                            */
    for(j=0; j<M; j++){
      bindata[j] = mean(arr+(j*(int)IND(binsize,i)), (int)IND(binsize,i));
    }
    /* Calculate the rms:                                          */
    IND(datarms,i) = rms(bindata, M);
    IND(rmserr,i)  = IND(datarms,i)/sqrt(2.0*M);

    /* Calculate extrapolated Gaussian-noise rms:                  */
    IND(gausserr,i) = stddata * sqrt(M/(IND(binsize,i)*(M - 1.0)));
  }

  /* Free arrays and return:                                       */
  free(bindata);
  free(arr);
  Py_XDECREF(size);
  return Py_BuildValue("[N,N,N,N]", datarms, rmserr, gausserr, binsize);
}


PyDoc_STRVAR(timeavg__doc__, "Residuals and chi-squared calculation.");

static PyMethodDef timeavg_methods[] = {
        {"binrms",  binrms, METH_VARARGS, binrms__doc__},
        {NULL,      NULL,   0,            NULL}
};

void inittimeavg(void){
  Py_InitModule3("timeavg", timeavg_methods, timeavg__doc__);
  import_array();
}

