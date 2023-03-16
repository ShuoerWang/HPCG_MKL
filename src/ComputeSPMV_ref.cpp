
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

#include <mkl.h>
/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/

int check_csr(int colidx[], int rowstr[]) {
    int valid_row = 1;
    int valid_col = 1;
    #pragma omp parallel for reduction(&&: valid_row)
    for (int i = 1; i < 1+1; i++) {
        valid_row = valid_row && (rowstr[i] < rowstr[i+1]);
    }

   
    if (!valid_row) {
        return 0;
    }

    #pragma omp parallel for reduction(&&: valid_col)
    for (int i = 1; i < 1+1; i++) {
        int valid_col_local = 1;
        for (int j = rowstr[i]; j < rowstr[i+1]-1; j++) {
            valid_col_local = valid_col_local && (colidx[j] < colidx[j+1]);
        }
        valid_col = valid_col && valid_col_local;
    }
    return valid_col;
}

int check_and_run(int colidx[], int rowstr[], double a[], double p[], double *q) {
    // if (!check_csr(colidx, rowstr)) {
    //     return 0;
    // }
    sparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 1, 1, rowstr, rowstr+1, colidx, a);

    double alpha = 1.0, beta = 0.0;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr, p, beta, q+1);

    mkl_sparse_destroy(A);
    return 1;
}

sparse_matrix_t check_and_convert_csr(const SparseMatrix & A, Vector & x, Vector & y) {
  int colidx[A.localNumberOfNonzeros];
  int rowstr[A.localNumberOfRows + 1];
  int values[A.localNumberOfNonzeros];

  int rowstr_idx = 0;
  rowstr[0] = 0;
  for (int i=0; i< A.localNumberOfRows; i++)  {
    rowstr[i+1] = rowstr[i] + A.nonzerosInRow[i];

    const double * const cur_vals = A.matrixValues[i];
    for (int j = 0; j< A.nonzerosInRow[i]; j++)
      values[rowstr[i] + j] = A.matrixValues[i][j];
      colidx[rowstr[i] + j] = A.mtxIndL[i][j];
  }

  check_and_run(colidx, rowstr, values, x, y);
}

int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
  return 0;
}