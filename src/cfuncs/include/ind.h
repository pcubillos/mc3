// Copyright (c) 2015-2016 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

/* Definitions for indexing Numpy arrays:                                   */

/* 1D double ndarray:                                                       */
#define INDd(a,i) *((double *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0)))
/* 1D integer ndarray:                                                      */
#define INDi(a,i) *((int    *)(PyArray_DATA(a) + i * PyArray_STRIDE(a, 0)))
