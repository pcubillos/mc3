// Copyright (c) 2015-2016 Patricio Cubillos and contributors.
// MC3 is open-source software under the MIT license (see LICENSE).

void daub4(double *a, const int n, const int isign) {
  /**********************************************************************
  Applies the Daubechies 4-coeficient wavelet filter to data vector
  a[0..n-1] (for isign=1) or it applies its transpose (for
  isign=-1).

  Parameters:
  -----------
  a:  Input data vector.
  n:  Hierarchy level of the transform.
  isign: If isign= 1, calculate DWT,
         If isign=-1, calculate the inverse DWT.

  Notes:
  ------
    This implementation follows the code from Numerical Recipes.
  **********************************************************************/
  const double C0 = 0.4829629131445341,
               C1 = 0.83651630373780772,
               C2 = 0.22414386804201339,
               C3 =-0.12940952255126034;
  int nh,
      i, j;    /* Auxilliary for-loop indices                       */
  double *dwt; /* The discreete wavelet transform                   */

  if (n<4)
    return;

  dwt = (double *)malloc(n *sizeof(double));
  nh = n>>1;
  if (isign >=0) {  /* Apply filter                                 */
    for(j=0, i=0; j<n-3; j+=2) {
      dwt[i   ] = C0*a[j] + C1*a[j+1] + C2*a[j+2] + C3*a[j+3];
      dwt[i+nh] = C3*a[j] - C2*a[j+1] + C1*a[j+2] - C0*a[j+3];
      i++;
    }
    dwt[i   ] = C0*a[n-2] + C1*a[n-1] + C2*a[0] + C3*a[1];
    dwt[i+nh] = C3*a[n-2] - C2*a[n-1] + C1*a[0] - C0*a[1];
  } else {          /* Apply transpose filter                       */
    dwt[0] = C2*a[nh-1] + C1*a[n-1] + C0*a[0] + C3*a[nh];
    dwt[1] = C3*a[nh-1] - C0*a[n-1] + C1*a[0] - C2*a[nh];
    for(i=0, j=2; i<nh-1; i++){
      dwt[j++] = C2*a[i] + C1*a[i+nh] + C0*a[i+1] + C3*a[i+nh+1];
      dwt[j++] = C3*a[i] - C0*a[i+nh] + C1*a[i+1] - C2*a[i+nh+1];
    }
  }
  /* Store values into input array:                                 */
  for (i=0; i<n; i++)
    a[i] = dwt[i];

  free(dwt);
  return;
}


void condition(double *a, const int n, const int isign){
  /******************************************************************
  Condition to make the modified rows of the 'detail filters' matrix
  return exactly zero when applied to smooth polynomial sequences
  like 1, 1, 1, 1, 1 or 1, 2, 3, 4, 5.

  Parameters:
  -----------
  a:  Input data vector.
  n:  Hierarchy level of the transform.
  isign: Do DWT for isign=1, or the inverse DWT for isign=-1.
  ******************************************************************/
  double t0, t1, t2, t3;
  if (n<4)
    return;
  if (isign >=0){
    t0 =  0.324894048898962*a[0]   + 0.0371580151158803*a[1];
    t1 =  1.00144540498130 *a[1];
    t2 =  1.08984305289504 *a[n-2];
    t3 = -0.800813234246437*a[n-2] + 2.09629288435324*a[n-1];
    a[0] = t0;
    a[1] = t1;
    a[2] = t2;
    a[3] = t3;
  } else {
    t0 = 3.07792649138669 *a[0]   - 0.114204567242137*a[1];
    t1 = 0.998556681198888*a[1];
    t2 = 0.917563310922261*a[n-2];
    t3 = 0.350522032550918*a[n-2] + 0.477032578540915*a[n-1];
    a[0]   = t0;
    a[1]   = t1;
    a[n-2] = t2;
    a[n-1] = t3;
  }
  return;
}


void dwt(double *a, int n, const int isign){
  /**********************************************************************
  One-dimensional discrete wavelet transform. This routine
  implements the pyramid algorithm, replacing a[0..n-1] by its
  wavelet (or inverse) transform.

  Parameters:
  -----------
  a:  Input data vector.
  n:  Length of the input vector.
  isign: If isign= 1, calculate DWT,
         If isign=-1, calculate the inverse DWT.

  Notes:
  ------
  With condition(...) commented out I get the same results as in IDL's
  built in DWT, So I'll keep it like that.
  **********************************************************************/
  int nn;
  if (n < 4)
    return;
  if (isign >= 0){
    //condition(a, n, 1);
    for(nn=n; nn>=4; nn>>=1)
      /* Start at largest hierarchy, and work toward smallest. */
      daub4(a, nn, isign);
  } else if (isign == -1){
    for(nn=4; nn<=n; nn<<=1)
      /* Start at smallest hierarchy, and work toward largest. */
      daub4(a, nn, isign);
  } else {
    for(nn=4; nn<=n; nn<<=1)
      /* Start at smallest hierarchy, and work toward largest. */
      daub4(a, nn, isign);
    //condition(a, n, -1);
  }
}
