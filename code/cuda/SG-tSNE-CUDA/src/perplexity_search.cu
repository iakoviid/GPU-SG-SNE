#include "perplexity_search.cuh"
#include <fstream>

template<class dataPoint>
void equalizeVertex(dataPoint*  val_P,
		    dataPoint * distances,
		    dataPoint perplexity,
		    int nn){

  bool found = false;
  dataPoint beta = 1.0;
  dataPoint min_beta = -std::numeric_limits<dataPoint>::max();
  dataPoint max_beta =  std::numeric_limits<dataPoint>::max();
  dataPoint tol = 1e-5;

  // Iterate until we found a good perplexity
  int iter = 0; dataPoint sum_P;
  while(!found && iter < 200) {

    // Compute Gaussian kernel row
    for(int m = 0; m < nn; m++) val_P[m] = exp(-beta * distances[m + 1]);

    // Compute entropy of current row
    sum_P = std::numeric_limits<dataPoint>::min();
    for(int m = 0; m < nn; m++) sum_P += val_P[m];
    dataPoint H = .0;
    for(int m = 0; m < nn; m++) H += beta * (distances[m + 1] * val_P[m]);
    H = (H / sum_P) + log(sum_P);

    // Evaluate whether the entropy is within the tolerance level
    dataPoint Hdiff = H - log(perplexity);
    if(Hdiff < tol && -Hdiff < tol) {
      found = true;
    }
    else {
      if(Hdiff > 0) {
	min_beta = beta;
	if(max_beta == std::numeric_limits<dataPoint>::max() || max_beta == -std::numeric_limits<dataPoint>::max())
	  beta *= 2.0;
	else
	  beta = (beta + max_beta) / 2.0;
      }
      else {
	max_beta = beta;
	if(min_beta == -std::numeric_limits<dataPoint>::max() || min_beta == std::numeric_limits<dataPoint>::max())
	  beta /= 2.0;
	else
	  beta = (beta + min_beta) / 2.0;
      }
    }

    // Update iteration counter
    iter++;
  }

  for(int m = 0; m < nn; m++) val_P[m] /= sum_P;

}

template<class dataPoint>
sparse_matrix<dataPoint> perplexityEqualization( int *I, dataPoint *D, int n, int nn, dataPoint u ){

  sparse_matrix<dataPoint> P;
  dataPoint *val;
  matidx *row, *col;

  // allocate space for CSC format
  val = static_cast<dataPoint *>( malloc( n*nn   * sizeof(dataPoint)) );
  row = static_cast<matidx *>( malloc( n*nn   * sizeof(matidx)) );
  col = static_cast<matidx *>( calloc( (n+1) , sizeof(matidx)) );

  // perplexity-equalization of kNN input
  for(int i = 0; i < n; i++) {

    equalizeVertex( &val[i*nn], &D[i*(nn+1)], u, nn );

  }

  // prepare column-wise kNN graph
  int nz = 0;
  for (int j=0; j<n; j++){
    col[j] = nz;
    for (int idx=0; idx<nn; idx++){
      row[nz + idx] = I[ j*(nn+1) + idx + 1 ];
    }
    nz += nn;
  }
  col[n] = nz;
	/*
	std::ofstream myfile;
  myfile.open("lord.txt");
	for(int i=0;i<n;i++){
		for(int j=col[i];j<col[i+1];j++){
			myfile<<row[j]<<"  "<<i<<"  "<<val[j]<<"\n";

		}
	}
	myfile.close();
*/
  if (nz != (nn*n) ) std::cerr << "Problem with kNN graph..." << std::endl;

  P.n   = n;
  P.m   = n;
  P.nnz = n * nn;
  P.row = row;
  P.col = col;
  P.val = val;

  return P;

}

template
sparse_matrix<float> perplexityEqualization( int *I, float *D, int n, int nn, float u );

template
sparse_matrix<double> perplexityEqualization( int *I, double *D, int n, int nn, double u );
