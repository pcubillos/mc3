// Copyright (c) 2015-2021 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "ind.h"
#include "stats.h"


PyDoc_STRVAR(binrms__doc__,
"Compute the binned root-mean-square and extrapolated           \n\
Gaussian-noise rms for a dataset.                               \n\
                                                                \n\
Parameters                                                      \n\
----------                                                      \n\
data: 1D ndarray                                                \n\
    A time-series dataset.                                      \n\
maxbins: Scalar                                                 \n\
    Maximum bin size to calculate.                              \n\
binstep: Integer                                                \n\
    Stepsize of binning indexing.                               \n\
                                                                \n\
Returns                                                         \n\
-------                                                         \n\
rms: 1D ndarray                                                 \n\
    RMS of binned data.                                         \n\
rmslo: 1D ndarray                                               \n\
    RMS lower uncertainties.                                    \n\
rmshi: 1D ndarray                                               \n\
    RMS upper uncertainties.                                    \n\
stderr: 1D ndarray                                              \n\
    Extrapolated RMS for Gaussian noise.                        \n\
binsz: 1D ndarray                                               \n\
    Bin sizes.                                                  \n\
                                                                \n\
Notes                                                           \n\
-----                                                           \n\
This function uses an asymptotic approximation to obtain the    \n\
rms uncertainties (rms_error = rms/sqrt(2M)) when the number of \n\
bins is M > 35.                                                 \n\
At smaller M, the errors become increasingly asymmetric. In this\n\
case the errors are numerically calculated from the posterior   \n\
PDF of the rms (an inverse-gamma distribution).                 \n\
See Cubillos et al. (2017), AJ, 153, 3.                         \n\
                                                                \n\
Uncredited developers                                           \n\
---------------------                                           \n\
Kevin Stevenson (UCF)                                           \n\
Matt Hardin (UCF)");

static PyObject *binrms(PyObject *self, PyObject *args){
  PyArrayObject *data,     /* Data array                           */
                *datarms,  /* RMS of the binned data               */
                *rmslo,    /* RMS low error bar                    */
                *rmshi,    /* RMS high error bar                   */
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
  int Mtemp;
  double s, ds, low, high;

  npy_intp size[1]; /* Size of output numpy array                  */

  /* Unpack arguments:                                             */
  if(!PyArg_ParseTuple(args, "O|ii", &data, &maxbins, &binstep)){
    return NULL;
  }
  /* Get data array size:                                          */
  dsize = (int)PyArray_DIM(data, 0);
  /* Set default maxbins:                                          */
  if (maxbins == -1)
    maxbins = dsize/2;

  /* Initialize numpy arrays:                                      */
  size[0] = (maxbins-1)/binstep + 1;
  datarms  = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
  rmslo    = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
  rmshi    = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
  gausserr = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
  binsize  = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

  /* Initialize pointers:                                          */
  bindata = (double *)malloc(dsize*sizeof(double));
  arr     = (double *)malloc(dsize*sizeof(double));
  for (i=0; i<dsize; i++)
    arr[i] = INDd(data,i);

  /* Calculate standard deviation of data:                         */
  stddata = std(arr, dsize);

  Mtemp = -1;  /* Flag to indicate when to recompute the IG CR.    */
  for(i=0; i<size[0]; i++){
    /* Set bin size and number of bins:                            */
    INDd(binsize,i) = 1 + i*binstep;
    M = dsize/(int)INDd(binsize,i);
    /* Bin the dataset:                                            */
    for(j=0; j<M; j++){
      bindata[j] = mean(arr+(j*(int)INDd(binsize,i)), (int)INDd(binsize,i));
    }
    /* Calculate the rms:                                          */
    INDd(datarms,i) = rms(bindata, M);
    INDd(rmslo,i) = INDd(rmshi,i) = INDd(datarms,i)/sqrt(2.0*M);

    /* Calculate extrapolated Gaussian-noise rms:                  */
    INDd(gausserr,i) = stddata * sqrt(M/(INDd(binsize,i)*(M - 1.0)));
    //INDd(gausserr,i) = stddata / sqrt(INDd(binsize,i));

    /* Large-bin-size regime:                                      */
    if (M <= 35){  /* Start computing at M=35, when RMSerror ~10%  */
      if (Mtemp != M){  /* Compute low, high only once for each M  */
        s  = INDd(gausserr,i);
        ds = INDd(gausserr,i) / sqrt(2.0*M);
        invgamma(M, s, ds, &low, &high);
        Mtemp = M;
      }
      /* Renormalize the error bars:                               */
      INDd(rmslo,i) = low *INDd(datarms,i)/INDd(gausserr,i);
      INDd(rmshi,i) = high*INDd(datarms,i)/INDd(gausserr,i);
    }
  }

  /* Free arrays and return:                                       */
  free(bindata);
  free(arr);
  Py_XDECREF(size);
  return Py_BuildValue("[N,N,N,N,N]", datarms, rmslo, rmshi, gausserr, binsize);
}


PyDoc_STRVAR(timeavg__doc__,"Time-averaging RMS curve with proper error bars.");

static PyMethodDef timeavg_methods[] = {
        {"binrms",  binrms, METH_VARARGS, binrms__doc__},
        {NULL,      NULL,   0,            NULL}
};

#if PY_MAJOR_VERSION >= 3
/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "timeavg",
    timeavg__doc__,
    -1,
    timeavg_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_timeavg (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else
/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void inittimeavg(void){
  Py_InitModule3("timeavg", timeavg_methods, timeavg__doc__);
  import_array();
}
#endif
