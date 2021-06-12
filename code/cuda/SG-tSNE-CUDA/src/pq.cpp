/*!
  \file   pq.cpp
  \brief

  <long description>

  \author Dimitris Floros
  \date   2019-06-28
*/


#include <iostream>
#include "pq.hpp"

void pq( coord       *       Fattr,
         coord       * const Y,
         matval const * const p_sp,
         matidx       *       ir,
         matidx       *       jc,
         int    const         n,
         int    const         d) {
  for(int i=0;i<n*d;i++){
    Fattr[i] = 0;
  }

  for (int j = 0; j < n; j++) {
  //for (unsigned int j = 0; j < n; j++) {

    coord accum[3] = {0};
    coord Yj[3];
    coord Yi[3];
    coord Ftemp[3];

    const int k = jc[j+1] - jc[j];    /* number of nonzero elements of each column */
    for(int i=0;i<d;i++){
      Yi[i]=Y[j*d+i];
    }
    /* for each non zero element */
    for (unsigned int idx = 0; idx < k; idx++) {

      const unsigned int i = (ir[jc[j] + idx]);

      for(int x=0;x<d;x++){
        Yj[x]=Y[i*d+x];
      }
      /* distance computation */
      coord dist=0;
      for(int x=0;x<d;x++){
        dist+= (Yj[x] - Yi[x])*(Yj[x] - Yi[x]);
      }

      /* P_{ij} \times Q_{ij} */
      const coord p_times_q = p_sp[jc[j]+idx] / (1+dist);
      for(int x=0;x<d;x++){
          Ftemp[x] = p_times_q * ( Yj[x] - Yi[x] );
          Fattr[ (i*d) + x ]+=Ftemp[x];
      }

    }

  }

}
