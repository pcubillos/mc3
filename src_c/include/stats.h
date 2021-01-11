// Copyright (c) 2015-2021 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

/******************************************************************
Calculate the mean value of the first n elements of data.

Parameters
----------
data: Pointer to array where to calculate the mean from.
n: Number of values to calculate the mean.

Returns
-------
datamean: The arithmetic mean.
******************************************************************/
double mean(double *data, const int n){
  int i;
  double datamean=0.0;
  for (i=0; i<n; i++)
    datamean += data[i];
  datamean /= n;
  return datamean;
}


/******************************************************************
Calculate the root mean square of the first n elements of data.

Parameters
----------
data: Pointer to array where to calculate the mean from.
n: Number of values to calculate the mean.

Returns
-------
datarms: The root mean square of the data.
******************************************************************/
double rms(double *data, const int n){
  int i;
  double datarms=0.0;
  for (i=0; i<n; i++)
    datarms += data[i]*data[i];
  datarms = sqrt(datarms/n);
  return datarms;
}


/******************************************************************
Calculate the standard deviation of the first n elements of data.

Parameters
----------
data: Pointer to array where to calculate the standard deviation from.
n: Number of values to calculate the mean.

Returns
-------
datastd: The standard deviation
******************************************************************/
double std(double *data, const int n){
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


/******************************************************************
Calculate the contribution of Jeffrey's and informative priors to
chi-squared:  sum{-2*ln(prior)}

Parameters
----------
prioroff: Parameter-prior difference.
priorlow: Lower uncertainty of an informative prior.
          A priorlow of -1 indicates a Jeffrey's prior.
priorup:  Upper uncertainty of an informative prior.

Returns
-------
chisq: -2 * sum of the logarithm of the priors.
******************************************************************/
double priors(PyArrayObject *prioroff, PyArrayObject *priorlow,
              PyArrayObject *priorup){
  int size, i;
  double chisq=0.0;
  size = (int)PyArray_DIM(prioroff, 0);

  for(i=0; i<size; i++){
    /* Jeffrey's prior:                                            */
    if (INDd(priorlow,i) == -1){
      chisq  += 2.0*log(INDd(prioroff,i));
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


/********************************************************************
Compute the inverse-gamma distribution credible-region error bars

The  distribution is given by:
  IG(x,M,s)  propto  1/x**M * exp(-M*s**2 / 2*x**2)

This is the marginal posterior PDF of a normal distribution with
standard deviation s and M data points.

Parameters
----------
M:    Number of datapoints from normal distribution.
s:    standard deviation of normal distribution
ds:   Asymptotically estimated error bar (large M)
low:  CR lower error bar (output).
high: CR upper error bar (output).

Notes
-----
The piece of code below here is heavily taylored to compute the CR
for the posterior distribution of a binned-rms sample.  Thus, on first
look, it may appear as a contraption.

Instead of calculating the PDF's values in ascending order, the code
computes the values in descending PDF-value order, i.e. starting at
x approx s, allowing the code to easily obtain the CR boundaries.
********************************************************************/
void invgamma(int M, double s, double ds, double *low, double *high){
  int i, ilo, ihi, n=10000;
  double psum=0.0, cdf=0.0;
  double xmin, xmax, dx, xlo, xhi, plo, phi, tmp;
  double *x, *posterior;

  x         = (double *)malloc(n*sizeof(double));
  posterior = (double *)malloc(n*sizeof(double));

  /* Posterior domain:                                             */
  xmax = s + 50.0*ds;
  xmin = s -  4.0*ds;
  if (xmin < 0.01*s)  /* Avoid x < 0.0                             */
    xmin = 0.01*s;
  dx = (xmax-xmin) / (n-1.0);

  /* Evaluate inverse-gamma PDF at their highest values:           */
  ilo = (int)((s-xmin)/dx);
  ihi = ilo + 1;
  xlo = xmin + ilo*dx;
  xhi = xmin + ihi*dx;
  plo = pow(xlo,-M) * exp(-M*s*s/(2*xlo*xlo));
  phi = pow(xhi,-M) * exp(-M*s*s/(2*xhi*xhi));

  /* Compute the PDF values in descending order:                   */
  for (i=0; i<n; i++){
    if (ilo < 0  || ihi >= n)
      break;
    if (plo > phi){
      posterior[i] = plo;       /* Take values                     */
      x[i] = xlo;
      xlo = xmin + (--ilo)*dx;  /* Update ilo                      */
      plo = pow(xlo,-M) * exp(-M*s*s/(2*xlo*xlo));
    }else{
      posterior[i] = phi;       /* Take values                     */
      x[i] = xhi;
      xhi = xmin + (++ihi)*dx;  /* Update ihi                      */
      phi = pow(xhi,-M) * exp(-M*s*s/(2*xhi*xhi));
    }
    psum += posterior[i];
  }
  /* Complete the sorted PDF:                                      */
  for (; i<n; i++){
    if (ilo < 0)
      x[i] = xmin + (ihi++)*dx;
    else
      x[i] = xmin + (ilo--)*dx;
    posterior[i] = pow(x[i],-M) * exp(-M*s*s/(2*x[i]*x[i]));
  }

  /* Normalize such that the sum equals 1.0                        */
  for (i=0; i<n; i++){
    posterior[i] = posterior[i] / psum;
  }

  /* Compute CDF and find the 68% percentile:                      */
  i = 0;
  while (cdf < 0.683){
    cdf += posterior[i++];
  }

  /* Get credible-region boundaries:                               */
  *low = x[i];
  *high = tmp = x[--i];
  if (*low > *high){
    *high = *low;
    *low  = tmp;
  }
  /* Loop until I get the extreme values:                          */
  while (1){
    tmp = x[--i];
    if (*low < tmp  && tmp < *high)
      break;
    else if (tmp < *low)
      *low = tmp;
    else
      *high = tmp;
  }
  /* Return the error-bar size instead of absolute value of the
     CR boundaries:                                                */
  *low  = (s-*low);
  *high = (*high-s);

  free(x);
  free(posterior);
}
