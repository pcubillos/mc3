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


double mean(double *data, const int n){
  /******************************************************************
  Calculate the mean value of the first n elements of data.

  Parameters:
  -----------
  data: Pointer to array where to calculate the mean from.
  n: Number of values to calculate the mean.

  Returns:
  --------
  datamean: The arithmetic mean.

  Modification History:
  ---------------------
  2014-05-15  patricio  Initial implementation.
  ******************************************************************/
  int i;
  double datamean=0.0;
  for (i=0; i<n; i++)
    datamean += data[i];
  datamean /= n;
  return datamean;
}


double rms(double *data, const int n){
  /******************************************************************
  Calculate the root mean square of the first n elements of data.

  Parameters:
  -----------
  data: Pointer to array where to calculate the mean from.
  n: Number of values to calculate the mean.

  Returns:
  --------
  datarms: The root mean square of the data.

  Modification History:
  ---------------------
  2014-05-15  patricio  Initial implementation.
  ******************************************************************/
  int i;
  double datarms=0.0;
  for (i=0; i<n; i++)
    datarms += data[i]*data[i];
  datarms = sqrt(datarms/n);
  return datarms;
}


double std(double *data, const int n){
  /******************************************************************
  Calculate the standard deviation of the first n elements of data.

  Parameters:
  -----------
  data: Pointer to array where to calculate the standard deviation from.
  n: Number of values to calculate the mean.

  Returns:
  --------
  datastd: The standard deviation

  Modification History:
  ---------------------
  2014-05-15  patricio  Initial implementation.
  ******************************************************************/
  int i;
  double datamean=0.0,
         datastd =0.0;
  double *zeromean;
  zeromean = (double *)malloc(n*sizeof(double));
  datamean = mean(data, n);
  for (i=0; i<n; i++)
    zeromean[i] = data[i] - datamean;
  datastd = rms(zeromean, n);
  free(zeromean);
  return datastd;
}


double priors(PyArrayObject *prioroff, PyArrayObject *priorlow,
              PyArrayObject *priorup, double *jchisq){
  /******************************************************************
  Calculate the contribution of Jeffrey's and informative priors to
  chi-squared:  sum{-2*ln(prior)}

  Parameters:
  -----------
  prioroff: Parameter-prior difference.
  priorlow: Lower uncertainty of an informative prior.
            A priorlow of -1 indicates a Jeffrey's prior.
  priorup:  Upper uncertainty of an informative prior.
  jchisq:   Jeffrey's contribution to chisq.

  Returns:
  --------
  chisq: -2 * sum of the logarithm of the priors.

  Modification History:
  ---------------------
  2015-04-15  patricio  Removed jchisq argument.
  2014-05-16  patricio  Initial implementation.
  ******************************************************************/
  int size, i;
  double chisq=0.0;
  *jchisq =0;
  size = PyArray_DIM(prioroff, 0);

  for(i=0; i<size; i++){
    /* Jeffrey's prior:                                            */
    if (INDd(priorlow,i) == -1){
      chisq   += 2.0*log(INDd(prioroff,i));
      *jchisq += 2.0*log(INDd(prioroff,i));
    }
    /* Informative prior:                                          */
    else if (INDd(prioroff,i) > 0){
      chisq += pow(INDd(prioroff,i)/INDd(priorup, i), 2);
    }else{
      chisq += pow(INDd(prioroff,i)/INDd(priorlow,i), 2);
    }
  }
  return chisq;
}


double recip2sum(double *data, int n){
  /******************************************************************
  Sum of the squared reciprocal data values

  Parameters:
  -----------
  data: array of values
  n: Number of elements to consider in the sum

  Returns:
  --------
  sum: The sum of the squared reciprocals

  Modification History:
  ---------------------
  2014-05-16  patricio  Initial implementation.
  ******************************************************************/
  int i;
  double sum=0.0;
  for (i=0; i<n; i++)
    sum += 1.0/(data[i]*data[i]);
  return sum;
}


double weightedsum(double *data, double *uncert, int n){
  /******************************************************************
  Weighted (by the squared reciprocal uncert) sum of data

  Parameters:
  -----------
  data:  Array of values
  uncert:  Data uncertainties
  n:  Number of elements to consider in the sum

  Returns:
  --------
  sum: The sum data weighted by the squared reciprocal of uncert

  Modification History:
  ---------------------
  2014-05-16  patricio  Initial implementation.
  ******************************************************************/
  int i;
  double sum=0.0;
  for (i=0; i<n; i++)
    sum += data[i]/(uncert[i]*uncert[i]);
  return sum;
}


void bindata(double *data, double *uncert, double *indp,
             int ndata, int binsize, PyArrayObject *bindata,
             PyArrayObject *binunc,  PyArrayObject *binindp){
  /******************************************************************
  Calculate the mean-weighted binned data, its standard deviation, and
  mean-binned indp

  Parameters:
  -----------
  data:    Array to calculate the weighted binned values
  uncert:  Data uncertainties
  indp:    Array to calculate the mean binned values
  ndata:   Number of data values
  binsize: Number of values per bin
  bindata: Array of weighted binned data (out)
  binunc:  Standard deviation of bindata (out)
  binindp: Array of mean binned indp (out)

  Modification History:
  ---------------------
  2014-05-16  patricio  Initial implementation.
  ******************************************************************/
  int nbins, start, i;

  /* Number of bins and binsize:                                   */
  nbins = PyArray_DIM(bindata, 0);
  for (i=0; i<nbins; i++){
    start = i*binsize;
    /* Mean of indp bins:                                          */
    INDd(binindp,i) = mean(indp+start, binsize);
    /* Standard deviation of the data mean:                        */
    INDd(binunc, i) = sqrt(1.0/recip2sum(uncert+start, binsize));
    /* Weighted mean of data bins:                                 */
    INDd(bindata,i) = weightedsum(data+start, uncert+start, binsize) *
                      INDd(binunc,i) * INDd(binunc,i);
  }
}
