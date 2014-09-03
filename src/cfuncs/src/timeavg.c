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
  if(!PyArg_ParseTuple(args, "O|dd", &data, &maxbins, &binstep)){
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
  return Py_BuildValue("[O,O,O,O]", datarms, rmserr, gausserr, binsize);
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

